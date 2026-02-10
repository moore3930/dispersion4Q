# build pairwise data from flores
import os
import shutil

def create_clean_dir(path):
    """
    Create a clean directory. If the directory exists, remove it first.
    :param path: Path of the directory to create.
    """
    # Remove the directory if it exists
    if os.path.exists(path):
        shutil.rmtree(path)
    # Create the directory
    os.makedirs(path)

lang_map = {"en": "eng_Latn", "de": "deu_Latn", "ar": "ace_Arab", "zh": "zho_Hans",
            "cs": "ces_Latn", "ru": "rus_Cyrl", "is": "isl_Latn"}

lang_map = {"gu": "guj_Gujr", "mr": "mar_Deva", "sd": "snd_Arab", "bn": "ben_Beng", "ur": "urd_Arab",
            "th": "tha_Thai", "bs": "bos_Latn", "my": "mya_Mymr", "xh": "xho_Latn", "km": "khm_Khmr",
            "hi": "hin_Deva", "lg": "lug_Latn", "ga": "gle_Latn", "jv": "jav_Latn", "cy": "cym_Latn",
            "pl": "pol_Latn", "ln": "lin_Latn", "wo": "wol_Latn", "tr": "tur_Latn", "kn": "kan_Knda",
            "bg": "bul_Cyrl", "be": "bel_Cyrl", "hy": "hye_Armn", "ta": "tam_Taml", "oc": "oci_Latn",
            "af": "afr_Latn", "ml": "mal_Mlym", "kk": "kaz_Cyrl", "tl": "tgl_Latn", "sr": "srp_Latn",
            "mk": "mkd_Cyrl", "sv": "swe_Latn", "ja": "jpn_Jpan", "en": "eng_Latn", "yo": "yor_Latn",
            "ig": "ibo_Latn", "pt": "por_Latn", "am": "amh_Ethi", "zu": "zul_Latn", "lt": "lit_Latn",
            "it": "ita_Latn", "lo": "lao_Laoo", "vi": "vie_Latn", "de": "deu_Latn", "fi": "fin_Latn",
            "uk": "ukr_Cyrl", "gl": "glg_Latn", "fr": "fra_Latn", "hr": "hrv_Latn", "so": "som_Latn",
            "is": "isl_Latn", "da": "dan_Latn", "sk": "slk_Latn", "id": "ind_Latn", "ha": "hau_Latn",
            "he": "heb_Hebr", "cs": "ces_Latn", "hu": "hun_Latn", "ru": "rus_Cyrl", "ka": "kat_Geor",
            "ko": "kor_Hang", "ar": "arb_Arab", "ast": "ast_Latn", "az": "azj_Latn", "ba": "bak_Cyrl",
            "ca": "cat_Latn", "ceb": "ceb_Latn", "el": "ell_Grek", "es": "spa_Latn", "et": "est_Latn",
            "fa": "pes_Arab", "ff": "fuv_Latn", "gd": "gla_Latn", "ht": "hat_Latn", "ilo": "ilo_Latn",
            "lb": "ltz_Latn", "lv": "lvs_Latn", "mg": "plt_Latn", "mn": "khk_Cyrl", "ms": "zsm_Latn",
            "ne": "npi_Deva", "nl": "nld_Latn", "no": "nno_Latn", "ns": "nso_Latn", "or": "ory_Orya",
            "pa": "pan_Guru", "ps": "pbt_Arab", "ro": "ron_Latn", "si": "sin_Sinh", "sl": "slv_Latn",
            "sq": "als_Latn", "ss": "ssw_Latn", "su": "sun_Latn", "sw": "swh_Latn", "tn": "tsn_Latn",
            "uz": "uzn_Latn", "yi": "ydd_Hebr", "zh": "zho_Hans", "mt": "mlt_Latn", "ti": "tir_Ethi",
            "ku": "kmr_Latn"}

lang_pairs = ["en-de", "de-en", "en-zh", "zh-en", "en-cs", "cs-en", "en-ru", "ru-en", "en-is", "is-en"]
# lang_pairs = ["en-de", "en-zh", "en-ru", "en-cs", "en-is"]
lang_pairs = ["en-de", "en-es", "en-ru", "en-zh", "en-fr", "en-nl", "en-it", "en-pt", "en-ko"]

test_dir = "test"

create_clean_dir(test_dir)

# get train data
for lang_pair in lang_pairs:
    src, tgt = lang_pair.split("-")
    src_lang = lang_map[src]
    tgt_lang = lang_map[tgt]

    # dev
    src_file = os.path.join("dev", "{}.dev".format(src_lang))
    tgt_file = os.path.join("dev", "{}.dev".format(tgt_lang))

    src_output_dir = os.path.join("test", "{}-{}".format(src, tgt))
    tgt_output_dir = os.path.join("test", "{}-{}".format(src, tgt))
    create_clean_dir(src_output_dir)
    create_clean_dir(tgt_output_dir)

    src_output = os.path.join(src_output_dir, "test.{}-{}.{}".format(src, tgt, src))
    tgt_output = os.path.join(tgt_output_dir, "test.{}-{}.{}".format(src, tgt, tgt))

    with open(src_file) as src_fin, open(tgt_file) as tgt_fin:
        with open(src_output, "a") as src_fout, open(tgt_output, "a") as tgt_fout:
            for src_sent, tgt_sent in zip(src_fin, tgt_fin):
                src_fout.write(src_sent)
                tgt_fout.write(tgt_sent)
            src_fout.flush()
            tgt_fout.flush()

    # devtest
    src_file = os.path.join("devtest", "{}.devtest".format(src_lang))
    tgt_file = os.path.join("devtest", "{}.devtest".format(tgt_lang))

    src_output_dir = os.path.join("test", "{}-{}".format(src, tgt))
    tgt_output_dir = os.path.join("test", "{}-{}".format(src, tgt))

    src_output = os.path.join(src_output_dir, "test.{}-{}.{}".format(src, tgt, src))
    tgt_output = os.path.join(tgt_output_dir, "test.{}-{}.{}".format(src, tgt, tgt))

    with open(src_file) as src_fin, open(tgt_file) as tgt_fin:
        with open(src_output, "a") as src_fout, open(tgt_output, "a") as tgt_fout:
            for src_sent, tgt_sent in zip(src_fin, tgt_fin):
                src_fout.write(src_sent)
                tgt_fout.write(tgt_sent)
            src_fout.flush()
            tgt_fout.flush()





