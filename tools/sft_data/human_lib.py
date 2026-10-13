"""human_lib.py: helpers of the human-written-only chat mix (see ForgeInfer - SFT Data Mix (human-written only) in the research notes).

Mechanical text transformations only (HTML to markdown, detokenisation, whitespace, templating): nothing here rewrites an answer with a model.  Quality rules for forum answers
(first-person narrative, references to comments / other answers / the site, edit markers) are simple regular expressions, listed in the manifest.
"""
import re
from html.parser import HTMLParser

import sft_lib as L  # noqa: E402  (the helpers of the first SFT build: clean, render, basic_checks, tokenizer, minhash ...)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# HTML (Stack Exchange bodies) -> plain markdown


_EMPTY_EM = re.compile(r"\*\*\s*\*\*|(?<![\w*])\*\s+\*(?!\*)|(?<![`\w])`\s*`(?![`\w])|~~\s*~~")


class _MD(HTMLParser):
    """Stack Exchange post bodies to markdown: paragraphs, lists, block quotes, fenced code (with the language hint of lang-xxx classes), inline code, bold, italics.  Links keep their text and lose the URL.
    Images, tables, frames and videos mark the post as unusable (`bad`): a text model cannot see them."""

    BAD = {"img", "table", "iframe", "video", "object", "embed", "svg", "canvas", "audio", "form", "input", "button", "math"}
    SKIP = {"script", "style"}
    BLOCK = {"p", "div", "section", "article", "header", "footer", "aside", "figure", "figcaption", "dl", "dt", "dd", "center"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tokens = []  # ("t", text) | ("b", n) newline events | ("li", marker, depth) | ("pre", lang, text) | ("bq", +1/-1)
        self.lists = []
        self.pre = False
        self.pre_buf = []
        self.pre_lang = ""
        self.code_in_line = False
        self.bad = None
        self.skip = 0
        self.in_pre_code = False
        self.links = 0  # <a href> elements whose URL was dropped

    # -- events
    def handle_starttag(self, tag, attrs):
        if tag in self.BAD:
            self.bad = self.bad or tag
            return
        if tag in self.SKIP:
            self.skip += 1
            return
        if self.pre:
            if tag == "code":
                self.in_pre_code = True
            elif tag == "br":
                self.pre_buf.append("\n")
            return
        if tag == "pre":
            self.pre, self.pre_buf, self.pre_lang = True, [], ""
            cls = dict(attrs).get("class") or ""
            m = re.search(r"lang-([A-Za-z0-9+#]+)", cls)
            if m and m.group(1) not in ("none", "default", "text"):
                self.pre_lang = m.group(1)
            return
        if tag in self.BLOCK:
            self.tokens.append(("b", 2))
        elif tag in ("ul", "ol"):
            self.tokens.append(("b", 1))
            self.lists.append([tag, 0])
        elif tag == "li":
            self.tokens.append(("b", 1))
            if self.lists:
                self.lists[-1][1] += 1
                marker = "- " if self.lists[-1][0] == "ul" else f"{self.lists[-1][1]}. "
            else:
                marker = "- "
            self.tokens.append(("li", marker, max(0, len(self.lists) - 1)))
        elif tag == "blockquote":
            self.tokens.append(("b", 2))
            self.tokens.append(("bq", 1))
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.tokens.append(("b", 2))
            self.tokens.append(("t", "**"))
        elif tag == "br":
            self.tokens.append(("b", 1))
        elif tag == "hr":
            self.tokens.append(("b", 2))
        elif tag in ("strong", "b"):
            self.tokens.append(("t", "**"))
        elif tag in ("em", "i"):
            self.tokens.append(("t", "*"))
        elif tag in ("del", "s", "strike"):
            self.tokens.append(("t", "~~"))
        elif tag == "a":
            self.links += 1
        elif tag == "code":
            self.tokens.append(("t", "`"))
            self.code_in_line = True
        elif tag == "sup":
            self.tokens.append(("t", "^"))
        elif tag == "sub":
            self.tokens.append(("t", "_"))

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
            return
        if tag in self.BAD:
            return
        if self.pre:
            if tag == "code":
                self.in_pre_code = False
            elif tag == "pre":
                self.pre = False
                text = "".join(self.pre_buf).strip("\n")
                self.tokens.append(("pre", self.pre_lang, text))
            return
        if tag in self.BLOCK:
            self.tokens.append(("b", 2))
        elif tag in ("ul", "ol"):
            if self.lists:
                self.lists.pop()
            self.tokens.append(("b", 1 if self.lists else 2))
        elif tag == "li":
            self.tokens.append(("b", 1))
        elif tag == "blockquote":
            self.tokens.append(("b", 2))
            self.tokens.append(("bq", -1))
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.tokens.append(("t", "**"))
            self.tokens.append(("b", 2))
        elif tag in ("strong", "b"):
            self.tokens.append(("t", "**"))
        elif tag in ("em", "i"):
            self.tokens.append(("t", "*"))
        elif tag in ("del", "s", "strike"):
            self.tokens.append(("t", "~~"))
        elif tag == "code":
            self.tokens.append(("t", "`"))
            self.code_in_line = False

    def handle_data(self, data):
        if self.skip:
            return
        if self.pre:
            self.pre_buf.append(data)
        else:
            self.tokens.append(("t", data))

    # -- rendering
    def render(self):
        lines, cur, bq = [], [], 0
        pending = None  # (marker, depth) for the first line of a list item

        def flush():
            nonlocal cur, pending
            text = re.sub(r"[ \t\r\f\v]+", " ", "".join(cur).replace("\n", " ")).strip()
            cur = []
            text = _EMPTY_EM.sub("", text).strip()  # empty bold / italic / code spans of ONE prose line (never inside a code block)
            if not text:
                return False
            if pending:
                text = "  " * pending[1] + pending[0] + text
                pending = None
            lines.append(("> " * bq) + text)
            return True

        def blank():
            if lines and lines[-1] != "":
                lines.append("")

        for tok in self.tokens:
            k = tok[0]
            if k == "t":
                cur.append(tok[1])
            elif k == "b":
                flush()
                if tok[1] == 2:
                    blank()
            elif k == "li":
                flush()
                pending = (tok[1], tok[2])
            elif k == "bq":
                flush()
                bq = max(0, bq + tok[1])
            elif k == "pre":
                flush()
                if lines and lines[-1] != "":
                    lines.append("")
                fence = "```" + tok[1]
                body = tok[2].replace("\r\n", "\n")
                for ln in [fence] + body.split("\n") + ["```"]:
                    lines.append(("> " * bq) + ln)
                lines.append("")
        flush()
        while lines and lines[-1] == "":
            lines.pop()
        text = "\n".join(lines)
        text = re.sub(r"\n{4,}", "\n\n\n", text)
        return text


_PREFIXES_BAD_HTML = re.compile(r"<img\b|<table\b|<iframe\b|<video\b", re.I)


def html_to_md(src):
    """(markdown, bad, links): bad is the first unusable tag (image, table, frame ...) or None; links = number of hyperlinks whose URL was dropped (the anchor text stays)."""
    if not src:
        return "", None, 0
    p = _MD()
    try:
        p.feed(src)
        p.close()
    except Exception:  # noqa: BLE001
        return "", "parse_error", 0
    return p.render(), p.bad, p.links


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# forum-answer rules (Stack Exchange, Reddit): an assistant must not talk about 'the comments', 'the OP', 'this site', nor tell its life story

_CODE_SPAN = re.compile(r"```.*?```|`[^`\n]*`", re.S)
_MATH_SPAN = re.compile(r"\$\$.*?\$\$|\$[^$\n]{1,200}\$", re.S)


def prose(text):
    """The text without code and without inline / display mathematics."""
    return _MATH_SPAN.sub(" ", _CODE_SPAN.sub(" ", text))


_META = re.compile(
    r"(?<![\w@])@[A-Za-z0-9_][A-Za-z0-9_.-]{2,}"  # @username
    r"|\b(?:down|up)[- ]?vot\w+|\baccepted answer\b|\b(?:my|this|that|your|his|her|their|another|previous|other|above|below|existing|earlier|top|accepted) answers?\b|\banswers? (?:above|below)\b"
    r"|\b(?:in|see|per|from|read|check) (?:the |my |your |his |her |their |these |those )?comments?\b|\bcomments? (?:above|below|here|there|section)\b|\b(?:as )?(?:mentioned|noted|said|stated|suggested|pointed out|explained|commented) (?:above|below|before|earlier|in the comments?)\b"
    r"|\b(?:stack ?exchange|stack ?overflow|superuser|serverfault|askubuntu|reddit|subreddit|r/[A-Za-z0-9_]+|moderators?|upvotes?|karma|this (?:site|forum|sub|thread|community)|(?:the )?question (?:above|below)|this question|your question title)\b"
    r"|\b(?:welcome to|thanks for (?:the )?(?:question|a2a)|great question|good question|nice question|interesting question)\b"
    r"|^\s*(?:edit|update|ps|p\.s\.|disclaimer|tl;?dr|source)\s*\d*\s*[:.\-]|\bedits?\s*\d*\s*:|\bthanks for (?:all )?the (?:up ?votes|gold|awards?|silver|replies|comments|support)\b|\brip my inbox\b"
    r"|\[(?:deleted|removed)\]|\bsee (?:the )?(?:image|screenshot|picture|photo|diagram|link|video) (?:below|above)|\b(?:screenshot|attachment)s?\b|\battached (?:image|file|picture|photo|screenshot|diagram)s?\b",
    re.I | re.M,
)
_META_CS = re.compile(r"\bOP(?:'s|’s)?\b|\bEDIT\b|\bUPDATED?\s*\d*\s*[:\-]|\bETA\b|\bIIRC\b|\bFWIW\b")  # case-sensitive: 'op' is a word of electronics ('op-amp')
# the first-person life story of a forum answer: relatives, jobs, years of experience (a chat assistant has none)
_NARRATIVE = re.compile(
    r"\bmy (?:wife|husband|girlfriend|boyfriend|partner|fianc\w+|mom|mum|mother|dad|father|son|daughter|kids?|children|baby|brother|sister|friends?|room-?mates?|flat-?mates?|boss|co-?workers?|colleagues?|grand\w+|uncle|aunt|neighbou?rs?|ex|professors?|advisors?|supervisors?|students?|dogs?|cats?|landlord|parents|family|team|company|school|university|lab|office|job|career|years?|experience|opinion|degree|phd|thesis|dissertation|country|city|town|hometown|apartment|house|home|garden|kitchen|car|truck|bike|laptop|computer|phone|server|setup|project|business|clients?|customers?|patients?|class(?:es)?|teacher|doctor|vet)\b"
    r"|\bas an? (?:[a-z]+ ){0,2}(?:myself,? )?(?:i|we)\b|\bi(?:'m| am| was| have been| work as| worked as) an? (?:\w+ ){0,3}(?:physicist|chemist|biologist|doctor|nurse|engineer|teacher|professor|lawyer|attorney|vet|veterinarian|historian|chef|baker|cook|student|phd|researcher|scientist|programmer|developer|mechanic|electrician|plumber|carpenter|contractor|pilot|photographer|musician|writer|editor|linguist|mathematician|economist|accountant|manager|recruiter|consultant|freelancer|sysadmin|architect|expert|native|grad|undergrad|hobbyist|amateur|beginner|newbie|veteran|owner|parent|mom|dad|father|mother)\b"
    r"|\bwhen i (?:was|worked|lived|moved|started|studied|taught|used to|first|got|had|did|went|bought|tried)\b|\bi (?:work|worked|teach|taught|study|studied|live|lived|grew up|moved|used to|have worked|have been (?:working|teaching|living|doing|using|playing|making|cooking|baking|building|brewing|riding|flying|travelling|traveling)|have (?:a|an|my) (?:degree|phd|master|bachelor)|bought|tried|tested|just (?:tried|bought|made|did)|went|ordered|got my|got a|got an)\b"
    r"|\b(?:last|this) (?:year|week|month|summer|winter|spring|fall|weekend)\b,? i\b|\byesterday,? i\b|\b(?:years|months|weeks|days) ago,? i\b|\bin my experience\b|\bin my opinion\b|\bpersonally,? i\b|\bhonestly,? i\b|\bto be honest,? i\b|\bi (?:swear|promise|guarantee)\b"
    r"|\b(?:i've|i have) (?:been|had|seen|heard|read|used|worked|tried|made|done|never|always|got|found|noticed|owned|lived|learned|taught|played)\b",
    re.I,
)
_FIRST = re.compile(r"\b(?:i|i'm|i’m|i've|i’ve|i'd|i’d|i'll|i’ll|me|my|mine|myself)\b", re.I)
_WORD_RE = re.compile(r"[A-Za-z']+")


def first_person_density(text):
    words = _WORD_RE.findall(prose(text))
    if not words:
        return 0.0
    return len(_FIRST.findall(prose(text))) / len(words)


def forum_answer_problem(ans, strict=False):
    """Reason to drop a forum answer (Stack Exchange / Reddit), or None.  strict = Reddit: the density limits are tighter."""
    pr = prose(ans)
    if _META.search(pr) or _META_CS.search(pr):
        return "meta_reference"
    n = len(_NARRATIVE.findall(pr))
    if n >= (1 if strict else 2):
        return "life_story"
    if first_person_density(ans) > (0.035 if strict else 0.05):
        return "first_person_dense"
    return None


_Q_BAD = re.compile(
    r"\[duplicate\]|\bbounty\b|\bclosed as\b|^\s*(?:edit|update)\s*\d*\s*[:\-]|\bEDIT\b|\bUPDATED?\s*\d*\s*[:\-]"
    r"|\b(?:see|as|per|in|from|says?|said|read|check|refer to|like) (?:the |my )?title\b|\btitle (?:says|already|is self[- ]explanatory|question)\b|\b(?:question|asked|asking) in the title\b|\bin the title\b"
    r"|\bsee (?:the )?(?:image|screenshot|picture|photo|diagram|link|video|below|above|attached)\b|\bas (?:you can )?see (?:in|on) the\b|\bscreenshots?\b|\battached (?:image|file|picture|photo|screenshot|diagram)s?\b"
    r"|\b(?:images?|pictures?|photos?|diagrams?) (?:below|above|attached)\b|\bthe (?:above|below) (?:image|picture|photo|diagram|code|error|question)\b|\b(?:link|links) (?:below|above)\b",
    re.I | re.M,
)


def forum_question_problem(q):
    pr = prose(q)
    if len(q) < 35 or len(q.split()) < 7:
        return "question_too_short"
    if _Q_BAD.search(pr):
        return "question_refers_elsewhere"
    return None


_TYPO = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'", "ʼ": "'", "“": '"', "”": '"', "„": '"', "–": "-", "—": "-", "−": "-", "…": "...", " ": " ", " ": " ", "​": ""})


def ascii_typography(s):
    """Typographic quotes, dashes, the ellipsis and no-break spaces to their ASCII forms (mechanical)."""
    return s.translate(_TYPO)


def english_ok_human(text):
    """L.english_ok on the prose only (code and mathematics removed first: LaTeX commands would count as non-stop words)."""
    pr = prose(text)
    if L.ascii_ratio(pr) < 0.90:
        return False
    words = re.findall(r"[a-z']+", pr.lower())
    if len(words) >= 25:
        stop = sum(1 for w in words if w in L._STOP)
        if stop / len(words) < 0.12:
            return False
    return True


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# WritingPrompts detokenisation (Fan et al. 2018 tokenised the prompts and partly the stories: "do n't", "`` ... ''", " ." , " \n ")

_DT = [
    (re.compile(r" ?\n ?\n ?"), "\n\n"),
    (re.compile(r" ?\n "), "\n"),
    (re.compile(r"`` ?"), '"'),
    (re.compile(r" ?''"), '"'),
    (re.compile(r"\s+(n't|N'T)\b"), r"\1"),
    (re.compile(r"\s+(['’])\s?(s|re|ve|ll|d|m|S|RE|VE|LL|D|M)\b"), r"\1\2"),
    (re.compile(r"\s+([.,!?;:%])(?=\s|$|[\"')\]])"), r"\1"),
    (re.compile(r"\(\s+"), "("),
    (re.compile(r"\s+\)"), ")"),
    (re.compile(r"\[\s+"), "["),
    (re.compile(r"\s+\]"), "]"),
    (re.compile(r"\s+\.\.\."), "..."),
]


def detokenize(s):
    s = s.replace("<newline>", "\n")
    for rx, rep in _DT:
        s = rx.sub(rep, s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


_ARTIFACT = re.compile(r" n't\b| ’ s\b| ' s\b|`` | ''|\s[.,!?;:]\s|\s[.,!?;]$| \) | \( ")


def tokenization_artifacts(text):
    return len(_ARTIFACT.findall(text))
