from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.trainers import BpeTrainer

from download_dataset import DATA_PATH, ensure_dataset_exists


TOKENIZER_PATH = Path(__file__).resolve().with_name("tokenizer.json")
VOCAB_SIZE = 2000
SPECIAL_TOKENS = [
    "[PAD]",
    "[UNK]",
    "[BOS]",
    "[EOS]",
]


def main() -> None:
    data_path = ensure_dataset_exists(DATA_PATH)

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.post_processor = ByteLevelProcessor(trim_offsets=True)

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )

    tokenizer.train([str(data_path)], trainer)
    tokenizer.save(str(TOKENIZER_PATH))

    print(f"Saved tokenizer to {TOKENIZER_PATH}")
    print(f"Final vocab size: {tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main()
