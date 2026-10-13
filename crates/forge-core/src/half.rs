//! IEEE binary16 and bfloat16 conversions (round to nearest even, subnormals, infinities, NaN), written from the bit layouts.

/// binary16 -> f32 (exact).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let man = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign << 31
        } else {
            // subnormal half: value = man * 2^-24; normalise it into an f32
            let mut e: i32 = 127 - 15 + 1;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            (sign << 31) | ((e as u32) << 23) | (m << 13)
        }
    } else if exp == 31 {
        (sign << 31) | 0x7F80_0000 | (man << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

/// f32 -> binary16, round to nearest even; overflow gives infinity.
pub fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xFF) as i32;
    let man = x & 0x7F_FFFF;
    if exp == 0xFF {
        // infinity or NaN (keep a quiet NaN payload bit)
        return sign
            | 0x7C00
            | if man != 0 {
                0x200 | (man >> 13) as u16
            } else {
                0
            };
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7C00;
    }
    if e <= 0 {
        // result is subnormal or zero
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000; // 24 significant bits
        let shift = (14 - e) as u32; // 13 + (1 - e)
        let q = m >> shift;
        let rem = m & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        let mut r = q as u16;
        if rem > halfway || (rem == halfway && (r & 1) == 1) {
            r += 1;
        }
        return sign | r;
    }
    let mut r = (((e as u32) << 10) | (man >> 13)) as u16;
    let rem = man & 0x1FFF;
    if rem > 0x1000 || (rem == 0x1000 && (r & 1) == 1) {
        r += 1; // a carry into the exponent is correct (it may become infinity)
    }
    sign | r
}

/// bfloat16 -> f32 (exact: a 16-bit left shift).
#[inline]
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// f32 -> bfloat16, round to nearest even (NaN stays NaN).
pub fn f32_to_bf16(f: f32) -> u16 {
    let x = f.to_bits();
    if (x & 0x7FFF_FFFF) > 0x7F80_0000 {
        return ((x >> 16) | 0x40) as u16;
    }
    let lsb = (x >> 16) & 1;
    (x.wrapping_add(0x7FFF + lsb) >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_round_trip_all_values() {
        for h in 0..=u16::MAX {
            let f = f16_to_f32(h);
            if f.is_nan() {
                assert!(f16_to_f32(f32_to_f16(f)).is_nan());
            } else {
                assert_eq!(f32_to_f16(f), h, "half {h:#06x} -> {f}");
            }
        }
    }

    #[test]
    fn f16_known_values() {
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0x7BFF), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f32_to_f16(65520.0), 0x7C00); // rounds up to infinity
        assert_eq!(f32_to_f16(65519.0), 0x7BFF);
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11)), 0x3C00); // tie to even
        assert_eq!(f32_to_f16(1.0 + 3.0 * 2f32.powi(-11)), 0x3C02); // tie to even (up)
        assert_eq!(f32_to_f16(2f32.powi(-25)), 0x0000); // tie to even -> 0
        assert_eq!(f32_to_f16(1.5 * 2f32.powi(-24)), 0x0002); // tie to even (up)
    }

    #[test]
    fn bf16_round_trip() {
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(f32_to_bf16(1.0), 0x3F80);
        assert_eq!(f32_to_bf16(1.0 + 2f32.powi(-8)), 0x3F80); // tie to even
        assert_eq!(f32_to_bf16(1.0 + 3.0 * 2f32.powi(-8)), 0x3F82);
        assert!(bf16_to_f32(f32_to_bf16(f32::NAN)).is_nan());
        assert_eq!(bf16_to_f32(f32_to_bf16(f32::INFINITY)), f32::INFINITY);
    }
}
