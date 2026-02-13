"""Reproduce gated-access failures for Unbabel/XCOMET-XXL."""

from comet.models import download_model


def test_xcomet_xxl_download():
    """
    Tries to download Unbabel/XCOMET-XXL.

    This will fail with the same stack trace seen in Slurm logs when:
    - token is missing/invalid for the node, or
    - the token account does not have access to the gated model.
    """
    download_model("Unbabel/XCOMET-XXL")


if __name__ == "__main__":
    test_xcomet_xxl_download()
