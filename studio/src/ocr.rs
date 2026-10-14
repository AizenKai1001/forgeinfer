//! Text from the pictures of a scanned PDF (a scan has no text layer). The pictures are taken out of the PDF here; on Windows the text recognition that comes with the system (`Windows.Media.Ocr`, for the
//! languages of the user's account) reads them: nothing is downloaded and nothing leaves the computer. Elsewhere there is nothing to read them with, and the reader says so.

use lopdf::Document;

/// A page's picture as a file that a decoder understands.
pub enum Picture {
    Jpeg(Vec<u8>),
    Png(Vec<u8>),
}

impl Picture {
    pub fn bytes(&self) -> &[u8] {
        match self {
            Picture::Jpeg(b) | Picture::Png(b) => b,
        }
    }
}

/// The biggest picture of every page of a PDF, in page order (a scan is one picture per page); None for a page without one or whose picture is in a format that is not read here (CCITT, JBIG2,
/// JPEG 2000, a palette, CMYK).
pub fn page_pictures(pdf: &[u8]) -> Result<Vec<Option<Picture>>, String> {
    let doc = Document::load_mem(pdf).map_err(|e| format!("not a PDF that can be read ({e})"))?;
    let mut out = Vec::new();
    for (_, page_id) in doc.get_pages() {
        let images = doc.get_page_images(page_id).unwrap_or_default();
        let best = images.iter().max_by_key(|i| i.width.max(0).saturating_mul(i.height.max(0)));
        out.push(best.and_then(|img| picture_of(&doc, img)));
    }
    Ok(out)
}

fn picture_of(doc: &Document, img: &lopdf::xobject::PdfImage) -> Option<Picture> {
    let filters = img.filters.clone().unwrap_or_default();
    // a JPEG is a file as it is
    if filters.len() == 1 && filters[0] == "DCTDecode" {
        return Some(Picture::Jpeg(img.content.to_vec()));
    }
    let data = doc.get_object(img.id).ok()?.as_stream().ok()?.decompressed_content().ok()?;
    let (w, h) = (usize::try_from(img.width).ok()?, usize::try_from(img.height).ok()?);
    if w == 0 || h == 0 || w > 20_000 || h > 20_000 {
        return None;
    }
    let bits = usize::try_from(img.bits_per_component.unwrap_or(8)).ok()?;
    let space = img.color_space.as_deref().unwrap_or("");
    let (color, depth, size) = match (space, bits) {
        ("DeviceRGB" | "CalRGB", 8) => (png::ColorType::Rgb, png::BitDepth::Eight, w * h * 3),
        ("DeviceGray" | "CalGray", 8) => (png::ColorType::Grayscale, png::BitDepth::Eight, w * h),
        ("DeviceGray" | "CalGray", 1) => (png::ColorType::Grayscale, png::BitDepth::One, h * w.div_ceil(8)),
        // a profile does not say how many components: the length of the data does
        ("ICCBased", 8) if data.len() >= w * h * 3 => (png::ColorType::Rgb, png::BitDepth::Eight, w * h * 3),
        ("ICCBased", 8) if data.len() >= w * h => (png::ColorType::Grayscale, png::BitDepth::Eight, w * h),
        _ => return None,
    };
    if data.len() < size {
        return None;
    }
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, w as u32, h as u32);
        enc.set_color(color);
        enc.set_depth(depth);
        enc.set_compression(png::Compression::Fast);
        let mut writer = enc.write_header().ok()?;
        writer.write_image_data(&data[..size]).ok()?;
    }
    Some(Picture::Png(out))
}

/// The text of a picture. Windows only.
#[cfg(windows)]
pub fn recognize(picture: &[u8]) -> Result<String, String> {
    use windows::Graphics::Imaging::{BitmapDecoder, BitmapPixelFormat, SoftwareBitmap};
    use windows::Media::Ocr::OcrEngine;
    use windows::Storage::Streams::{DataWriter, InMemoryRandomAccessStream};
    fn e(err: windows::core::Error) -> String {
        err.message()
    }
    let engine = OcrEngine::TryCreateFromUserProfileLanguages().map_err(|_| "Windows has no text recognition for the languages of this account (Settings > Time & language > Language: add one with \"Optical character recognition\")".to_string())?;
    let stream = InMemoryRandomAccessStream::new().map_err(e)?;
    let writer = DataWriter::CreateDataWriter(&stream).map_err(e)?;
    writer.WriteBytes(picture).map_err(e)?;
    writer.StoreAsync().map_err(e)?.join().map_err(e)?;
    writer.DetachStream().map_err(e)?;
    stream.Seek(0).map_err(e)?;
    let decoder = BitmapDecoder::CreateAsync(&stream).map_err(e)?.join().map_err(e)?;
    let bitmap = decoder.GetSoftwareBitmapAsync().map_err(e)?.join().map_err(e)?;
    let bitmap = SoftwareBitmap::Convert(&bitmap, BitmapPixelFormat::Bgra8).map_err(e)?;
    let max = OcrEngine::MaxImageDimension().unwrap_or(2600) as i32;
    if bitmap.PixelWidth().map_err(e)? > max || bitmap.PixelHeight().map_err(e)? > max {
        return Err(format!("the picture is larger than the {max} pixels that Windows text recognition reads"));
    }
    let result = engine.RecognizeAsync(&bitmap).map_err(e)?.join().map_err(e)?;
    // one line of the page per line of text (the result's own `Text` runs them together)
    let mut text = String::new();
    for line in result.Lines().map_err(e)? {
        text.push_str(&line.Text().map_err(e)?.to_string());
        text.push('\n');
    }
    Ok(text)
}

#[cfg(not(windows))]
pub fn recognize(_picture: &[u8]) -> Result<String, String> {
    Err("text recognition from pictures is built into Windows only".to_string())
}

/// Fill the pages of a PDF that have no text (`pages`, one string per page) with the text read from their pictures: a scan is all such pages, a report with a scanned appendix some of them. Returns
/// why nothing could be read when no page got any text.
pub fn read_blank_pages(pdf: &[u8], pages: &mut [String]) -> Result<(), String> {
    let blank: Vec<usize> = pages.iter().enumerate().filter(|(_, p)| p.trim().is_empty()).map(|(i, _)| i).collect();
    if blank.is_empty() {
        return Ok(());
    }
    let pictures = page_pictures(pdf)?;
    if blank.iter().all(|&i| pictures.get(i).is_none_or(Option::is_none)) {
        return Err("its pages are pictures in a format that is not read here".to_string());
    }
    let (mut read, mut first_error) = (0, None);
    for i in blank {
        if let Some(Some(p)) = pictures.get(i) {
            match recognize(p.bytes()) {
                Ok(t) => {
                    read += usize::from(!t.trim().is_empty());
                    pages[i] = t;
                }
                Err(e) => {
                    first_error.get_or_insert(e);
                }
            }
        }
    }
    match first_error {
        Some(e) if read == 0 => Err(e),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A one-page PDF whose page is a picture: `data` is the stream, `extra` its dictionary entries.
    fn picture_pdf(extra: &str, data: &[u8]) -> Vec<u8> {
        let mut pdf = b"%PDF-1.4\n".to_vec();
        let mut offsets = Vec::new();
        let objs: Vec<Vec<u8>> = vec![
            b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Contents 4 0 R /Resources << /XObject << /Im0 5 0 R >> >> >>".to_vec(),
            {
                let c = b"q 100 0 0 100 0 0 cm /Im0 Do Q";
                let mut v = format!("<< /Length {} >>\nstream\n", c.len()).into_bytes();
                v.extend_from_slice(c);
                v.extend_from_slice(b"\nendstream");
                v
            },
            {
                let mut v = format!("<< /Type /XObject /Subtype /Image {extra} /Length {} >>\nstream\n", data.len()).into_bytes();
                v.extend_from_slice(data);
                v.extend_from_slice(b"\nendstream");
                v
            },
        ];
        for (i, o) in objs.iter().enumerate() {
            offsets.push(pdf.len());
            pdf.extend(format!("{} 0 obj\n", i + 1).bytes());
            pdf.extend_from_slice(o);
            pdf.extend_from_slice(b"\nendobj\n");
        }
        let xref = pdf.len();
        pdf.extend(format!("xref\n0 {}\n0000000000 65535 f \n", objs.len() + 1).bytes());
        for off in offsets {
            pdf.extend(format!("{off:010} 00000 n \n").bytes());
        }
        pdf.extend(format!("trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n", objs.len() + 1).bytes());
        pdf
    }

    #[test]
    fn the_picture_of_a_page_comes_out_as_a_file_a_decoder_reads() {
        // a JPEG is taken as it is
        let jpeg = picture_pdf("/Width 4 /Height 4 /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode", b"\xff\xd8\xff\xe0fake jpeg\xff\xd9");
        let pages = page_pictures(&jpeg).unwrap();
        assert!(matches!(&pages[0], Some(Picture::Jpeg(b)) if b.starts_with(&[0xff, 0xd8])));
        // raw 8-bit pixels become a PNG that has them back: 2 x 2 RGB
        let px: Vec<u8> = (0..12).map(|i| i * 20).collect();
        let raw = picture_pdf("/Width 2 /Height 2 /ColorSpace /DeviceRGB /BitsPerComponent 8", &px);
        let png_bytes = match &page_pictures(&raw).unwrap()[0] {
            Some(Picture::Png(b)) => b.clone(),
            _ => panic!("a PNG was expected"),
        };
        let mut reader = png::Decoder::new(std::io::Cursor::new(png_bytes)).read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).unwrap();
        assert_eq!((info.width, info.height, info.color_type), (2, 2, png::ColorType::Rgb));
        assert_eq!(&buf[..12], &px[..]);
        // a bilevel scan: 10 x 2 pixels, rows padded to whole bytes
        let bits = picture_pdf("/Width 10 /Height 2 /ColorSpace /DeviceGray /BitsPerComponent 1", &[0xff, 0xc0, 0x00, 0x00]);
        assert!(matches!(&page_pictures(&bits).unwrap()[0], Some(Picture::Png(_))));
        // formats that are not read: a fax picture, a palette, data that is too short, no picture
        for extra in ["/Width 8 /Height 8 /ColorSpace /DeviceGray /BitsPerComponent 1 /Filter /CCITTFaxDecode", "/Width 2 /Height 2 /ColorSpace [/Indexed /DeviceRGB 1 <000000ffffff>] /BitsPerComponent 8"] {
            assert!(page_pictures(&picture_pdf(extra, &[0, 1, 2, 3])).unwrap()[0].is_none(), "{extra}");
        }
        assert!(page_pictures(&picture_pdf("/Width 100 /Height 100 /ColorSpace /DeviceRGB /BitsPerComponent 8", &[0; 10])).unwrap()[0].is_none());
        assert!(page_pictures(b"not a pdf").is_err());
        let mut pages = vec![String::new()];
        assert!(read_blank_pages(&picture_pdf("/Width 8 /Height 8 /ColorSpace /DeviceGray /BitsPerComponent 1 /Filter /CCITTFaxDecode", &[0; 8]), &mut pages).unwrap_err().contains("not read here"));
        // nothing to fill: nothing is read
        let mut full = vec!["text".to_string()];
        assert!(read_blank_pages(b"not even a pdf", &mut full).is_ok() && full[0] == "text");
    }

    /// The text recognition of Windows, on a picture of text drawn here (a bitmap font would be too long: this uses the scans on the disk): FORGE_TEST_SCANS=file;file cargo test --lib live_ocr -- --ignored --nocapture
    #[test]
    #[ignore]
    fn live_ocr() {
        for p in std::env::var("FORGE_TEST_SCANS").unwrap_or_default().split(';').filter(|p| !p.is_empty()) {
            let t0 = std::time::Instant::now();
            let pdf = std::fs::read(p).unwrap();
            let mut pages = vec![String::new(); Document::load_mem(&pdf).unwrap().get_pages().len()];
            match read_blank_pages(&pdf, &mut pages).map(|_| pages.join("\u{c}")) {
                Ok(text) => {
                    let pages: Vec<&str> = text.split('\u{c}').collect();
                    // (only figures are printed: a scan is somebody's paper; FORGE_TEST_SHOW=80 adds the start of page 1)
                    let letters = text.chars().filter(|c| c.is_alphabetic()).count();
                    let show = std::env::var("FORGE_TEST_SHOW").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0);
                    println!("{p}: {} pages, {} characters ({} letters, {} words) in {:.1} s{}", pages.len(), text.chars().count(), letters, text.split_whitespace().count(), t0.elapsed().as_secs_f32(), if show > 0 { format!("; page 1 starts: {:?}", pages[0].chars().take(show).collect::<String>()) } else { String::new() });
                }
                Err(e) => println!("{p}: ERROR {e}"),
            }
        }
    }
}
