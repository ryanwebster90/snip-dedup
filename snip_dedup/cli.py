"""cli entry point"""

from snip_dedup.snip_download import snip_download
import fire


def main():
    """Main entry point"""
    fire.Fire(
        {
            "download": snip_download,
        }
    )


if __name__ == "__main__":
    main()
