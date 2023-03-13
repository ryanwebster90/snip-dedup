"""cli entry point"""

from snip_dedup.snip_download import snip_download
from snip_dedup.snip_compress import snip_compress
from snip_dedup.snip_index import snip_index
import fire


def main():
    """Main entry point"""
    fire.Fire(
        {
            "download": snip_download,
            "compress": snip_compress,
            "index": snip_index,
        }
    )


if __name__ == "__main__":
    main()
