"""human_cfg.py: configuration of the human-written-only chat mix (see ForgeInfer - SFT Data Mix (human-written only) in the research notes).

Everything that is a choice (which sites, which subreddits, which files, how many tokens per source, licenses and provenance quotes) lives here, so that the downloader (fetch_human.py), the
pool builder (human_pools.py), the assembler (human_assemble.py) and the manifest writer (human_manifest.py) agree.
"""
import collections
import os

WORK = "/mnt/data/forgeinfer-data/sft_work_human"
RAW = os.path.join(WORK, "raw")
POOL = os.path.join(WORK, "pool")
CARDS = os.path.join(WORK, "cards")
OLD_RAW = "/mnt/data/forgeinfer-data/sft_raw"  # raw files of the first SFT build (oasst2, dolly, gsm8k, hendrycks_math, the benchmark test sets): read only
OLD_WORK = "/mnt/data/forgeinfer-data/sft_work"  # the first build's pools and the identity pool (read only)
OUT = "/mnt/data/forgeinfer-data/tokens_sft_human"
OUT_REPLAY = "/mnt/data/forgeinfer-data/tokens_sft_human_replay"
BUILD = os.environ.get("HUMAN_BUILD", os.path.join(WORK, "build"))

SE_REPO = "HuggingFaceH4/stack-exchange-preferences"
# Stack Exchange: site -> (weight in the Stack Exchange budget, number of parquet files to download (None = all files of the site)).  The file names are data/<site>/train-0000k-of-0000n.parquet.
# No religion / politics / parenting sites (opinions, personal narratives), no non-English language-learning sites, no meta sites, no vendor-specific CMS sites.
SE_SITES = collections.OrderedDict([
    # science and engineering
    ("physics.stackexchange.com", (3.0, None)), ("chemistry.stackexchange.com", (1.4, None)), ("biology.stackexchange.com", (1.2, None)), ("astronomy.stackexchange.com", (0.7, None)),
    ("space.stackexchange.com", (0.8, None)), ("earthscience.stackexchange.com", (0.3, None)), ("engineering.stackexchange.com", (0.5, None)), ("electronics.stackexchange.com", (2.0, None)),
    ("mechanics.stackexchange.com", (0.8, None)), ("stats.stackexchange.com", (1.6, None)), ("cs.stackexchange.com", (0.9, None)), ("datascience.stackexchange.com", (0.5, None)),
    ("economics.stackexchange.com", (0.5, None)), ("cogsci.stackexchange.com", (0.2, None)), ("linguistics.stackexchange.com", (0.5, None)), ("skeptics.stackexchange.com", (0.6, None)),
    ("quantumcomputing.stackexchange.com", (0.2, None)), ("ai.stackexchange.com", (0.25, None)), ("robotics.stackexchange.com", (0.2, None)), ("materials.stackexchange.com", (0.15, None)),
    ("sound.stackexchange.com", (0.3, None)), ("dsp.stackexchange.com", (0.3, None)),
    # mathematics
    ("math.stackexchange.com", (4.0, 4)), ("matheducators.stackexchange.com", (0.3, None)),
    # language, humanities, culture
    ("english.stackexchange.com", (2.5, None)), ("ell.stackexchange.com", (1.6, None)), ("history.stackexchange.com", (1.1, None)), ("philosophy.stackexchange.com", (1.3, None)),
    ("writers.stackexchange.com", (1.3, None)), ("literature.stackexchange.com", (0.3, None)), ("movies.stackexchange.com", (0.7, None)), ("music.stackexchange.com", (1.0, None)),
    ("scifi.stackexchange.com", (1.0, None)), ("worldbuilding.stackexchange.com", (1.0, None)), ("rpg.stackexchange.com", (0.7, None)), ("gaming.stackexchange.com", (0.8, None)),
    ("puzzling.stackexchange.com", (0.6, None)), ("chess.stackexchange.com", (0.3, None)), ("boardgames.stackexchange.com", (0.3, None)), ("anime.stackexchange.com", (0.2, None)),
    ("mythology.stackexchange.com", (0.1, None)),
    # practical life and how-to
    ("cooking.stackexchange.com", (1.6, None)), ("diy.stackexchange.com", (2.2, None)), ("gardening.stackexchange.com", (0.7, None)), ("travel.stackexchange.com", (1.6, None)),
    ("money.stackexchange.com", (1.3, None)), ("workplace.stackexchange.com", (1.6, None)), ("academia.stackexchange.com", (1.3, None)), ("lifehacks.stackexchange.com", (0.15, None)),
    ("outdoors.stackexchange.com", (0.4, None)), ("bicycles.stackexchange.com", (0.7, None)), ("aviation.stackexchange.com", (0.7, None)), ("fitness.stackexchange.com", (0.4, None)),
    ("pets.stackexchange.com", (0.3, None)), ("photo.stackexchange.com", (0.9, None)), ("graphicdesign.stackexchange.com", (0.7, None)), ("ux.stackexchange.com", (0.7, None)),
    ("law.stackexchange.com", (0.4, None)), ("interpersonal.stackexchange.com", (0.4, None)), ("woodworking.stackexchange.com", (0.15, None)), ("crafts.stackexchange.com", (0.1, None)),
    ("coffee.stackexchange.com", (0.05, None)), ("beer.stackexchange.com", (0.05, None)), ("homebrew.stackexchange.com", (0.1, None)), ("martialarts.stackexchange.com", (0.1, None)),
    ("sports.stackexchange.com", (0.1, None)), ("3dprinting.stackexchange.com", (0.15, None)), ("freelancing.stackexchange.com", (0.1, None)), ("expatriates.stackexchange.com", (0.1, None)),
    ("sustainability.stackexchange.com", (0.1, None)), ("opensource.stackexchange.com", (0.15, None)), ("pm.stackexchange.com", (0.3, None)), ("softwarerecs.stackexchange.com", (0.2, None)),
    ("hardwarerecs.stackexchange.com", (0.1, None)),
    # computing
    ("unix.stackexchange.com", (2.5, None)), ("superuser.com", (2.2, None)), ("askubuntu.com", (1.6, None)), ("serverfault.com", (1.1, None)), ("softwareengineering.stackexchange.com", (1.6, None)),
    ("codereview.stackexchange.com", (1.0, None)), ("security.stackexchange.com", (1.0, None)), ("crypto.stackexchange.com", (0.5, None)), ("apple.stackexchange.com", (0.9, None)),
    ("android.stackexchange.com", (0.4, None)), ("webmasters.stackexchange.com", (0.4, None)), ("networkengineering.stackexchange.com", (0.4, None)), ("dba.stackexchange.com", (0.6, None)),
    ("raspberrypi.stackexchange.com", (0.4, None)), ("arduino.stackexchange.com", (0.4, None)), ("blender.stackexchange.com", (0.3, None)), ("gamedev.stackexchange.com", (0.5, None)),
    ("tex.stackexchange.com", (0.6, 2)), ("retrocomputing.stackexchange.com", (0.3, None)), ("vi.stackexchange.com", (0.15, None)), ("emacs.stackexchange.com", (0.15, None)),
    ("Stackoverflow.com", (3.0, 5)),
])
SE_DATE_MAX = "2022/06/30"  # question date: every answer was written before the public release of ChatGPT (2022-11-30) with a five-month margin

SHP_REPO = "stanfordnlp/SHP"
SHP_SUBS = collections.OrderedDict([  # subreddit -> weight in the SHP budget
    ("askscience", 3.0), ("askhistorians", 1.2), ("askphysics", 1.0), ("askphilosophy", 1.5), ("explainlikeimfive", 3.0), ("askculinary", 2.0), ("askbaking", 1.2), ("askengineers", 2.0),
    ("askacademia", 1.5), ("asksocialscience", 0.4), ("askanthropology", 0.4), ("asksciencefiction", 1.0), ("askcarguys", 0.3), ("askhr", 0.8),
])
SHP_TS_MAX = 1661990400  # 2022-09-01 00:00 UTC: comments written before the release of ChatGPT with a three-month margin

# small sources to download: (repo, [file patterns relative to the repo root]); the patterns are matched against the hub file list
SMALL = [
    ("HuggingFaceH4/no_robots", ["data/train-*.parquet", "data/test-*.parquet"]),
    ("CohereLabs/aya_dataset", ["data/train-*.parquet", "data/test-*.parquet"]),
    ("deepmind/aqua_rat", ["raw/train-*.parquet", "raw/validation-*.parquet"]),
    ("codeparrot/apps", ["train.jsonl"]),
    ("google-research-datasets/mbpp", ["full/train-*.parquet", "full/validation-*.parquet", "full/test-*.parquet"]),
    ("christopher/rosetta-code", ["data/train-*.parquet"]),
    ("rajpurkar/squad", ["plain_text/train-*.parquet"]),
    ("stanfordnlp/coqa", ["data/train-*.parquet"]),
    ("google-research-datasets/nq_open", ["nq_open/train-*.parquet"]),
    ("mandarjoshi/trivia_qa", ["rc.nocontext/train-*.parquet"]),
    ("abisee/cnn_dailymail", ["3.0.0/train-00000-of-00003.parquet"]),
    ("euclaise/writingprompts", ["data/train-00000-of-00002-*.parquet"]),
    ("knkarthick/dialogsum", ["train.csv"]),
    ("code-search-net/code_search_net", ["python/train-*.parquet"]),
]
