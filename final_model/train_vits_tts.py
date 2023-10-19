import os
from glob import glob

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import CharactersConfig, Vits, VitsArgs, VitsAudioConfig
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

output_path = os.path.dirname(os.path.abspath(__file__))
root = "/content/drive/MyDrive/finaltrainbytookk/TTS/recipes/multilingual/vits_tts/"
mailabs_path = "/content/drive/MyDrive/finaltrainbytookk/TTS/recipes/multilingual/vits_tts/mailabs/**"
dataset_paths = glob(mailabs_path)
dataset_config = [
    BaseDatasetConfig(formatter="mailabs", meta_file_train=None,
                      path=path, language=path.split("/")[-1])
    for path in dataset_paths
]

audio_config = VitsAudioConfig(
    sample_rate=22050,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    mel_fmin=0,
    mel_fmax=None,
)

vitsArgs = VitsArgs(
    use_language_embedding=True,
    embedded_language_dim=4,
    use_speaker_embedding=True,
    speaker_encoder_config_path=os.path.join(root, "config_se.json"),
    speaker_encoder_model_path=os.path.join(root, "model_se.pth"),
    use_sdp=False,
)

config = VitsConfig(
    model_args=vitsArgs,
    audio=audio_config,
    run_name="vits_vctk",
    use_speaker_embedding=True,
    batch_size=32,
    eval_batch_size=16,
    batch_group_size=0,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="multilingual_cleaners",
    use_phonemes=False,
    phoneme_language="en-us",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    compute_input_seq_cache=True,
    print_step=25,
    use_language_weighted_sampler=True,
    print_eval=False,
    mixed_precision=False,
    min_audio_len=32 * 256 * 4,
    max_audio_len=160000,
    output_path=output_path,
    datasets=dataset_config,
    characters=CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="<PAD>",
        eos="<EOS>",
        bos="<BOS>",
        blank="<BLNK>",
        characters="!¡'(),-.:;¿?abcdefghijklmnopqrstuvwxyzµßàáâäåæçèéêëìíîïñòóôöùúûüąćęłńœśşźżƒабвгдежзийклмнопрстуфхцчшщъыьэюяёєіїґӧ «°±µ»$%&‘’‚“`”„ກຂຄງຈສຊຍດຕຖທນບປຜຝພຟມຢຣລວຫອຮະັົາໍຳິີຶືຸູເແໂໃໄໆ່້໋໊ໜໝຯຽ໌ຼ",
        punctuations="!¡'(),-.:;¿? ",
        phonemes=None,
    ),
    test_sentences=[
        [
            "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
            "Linda",
            None,
            "en",
        ],
        [
            "ການທົດລອງຂຸດຄົ້ນ",
            "Nick",
            None,
            "lo-la"
        ],
                [
            "ແຮ່ທາດເປັນນະໂຍບາຍ",
            "Nick",
            None,
            "lo-la"
        ],
                [
            "ຂໍ້ຂ້ອງໃຈຂອງຜູ້ແທນ",
            "Nick",
            None,
            "lo-la"
        ],
                [
            "ມີວົງແຫວນຢູ່ກ້ອງດອກເຫັດ",
            "Nick",
            None,
            "lo-la"
        ],
                [
            "ເພື່ອເປັນແບບຢ່າງແລະເປັນລະບຽບ",
            "Nick",
            None,
            "lo-la"
        ],
                [
            "ຈັດຕັ້ງການຄົ້ນຄວ້າແລກປ່ຽນບົດຮຽນ",
            "Nick",
            None,
            "lo-la"
        ],
                [
            "ນີ້ມັນບໍ່ຕ່າງຫຍັງກັບການເພີ່ມ",
            "Nick",
            None,
            "lo-la"
        ]
    ],
)

# force the convertion of the custom characters to a config attribute
config.from_dict(config.to_dict())

# init audio processor
ap = AudioProcessor(**config.audio.to_dict())

# load training samples
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# init speaker manager for multi-speaker training
# it maps speaker-id to speaker-name in the model and data-loader
speaker_manager = SpeakerManager()
speaker_manager.set_ids_from_data(
    train_samples + eval_samples, parse_key="speaker_name")
config.model_args.num_speakers = speaker_manager.num_speakers

language_manager = LanguageManager(config=config)
config.model_args.num_languages = language_manager.num_languages

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# config is updated with the default characters if not defined in the config.
tokenizer, config = TTSTokenizer.init_from_config(config)

# init model
model = Vits(config, ap, tokenizer, speaker_manager, language_manager)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)
trainer.fit()
