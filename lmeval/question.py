from pydantic import Field
from pathlib import Path
from hashlib import blake2b
from typing import List, Dict

from collections import defaultdict
from .custom_model import CustomModel
from .media import Media
from .models import LMAnswer
from .enums import Modality, FileType


class QuestionSource(CustomModel):
    name: str
    url: str = Field(default="")
    description: str = Field(default="")

    def __str__(self) -> str:
        return str(self.name)


class Question(CustomModel):
    id: int  # we need for (de)serialization
    question: str
    language: str = Field(default="en")

    # answer
    answer: str

    # multiple choice questions only
    # Added at benchmark generation time
    choices: List[str] = Field(default_factory=list)  # alternative choices for multiple choice questions
    # dynamically computed at first prompt rendering time we keep track per prompt_version hence the dicts
    answer_letter: str = Field(default="")
    letters: str = Field(default="")
    multi_choices: str = Field(default="")

    # cache template rendering keyed by prompt version to ensure consistency
    # accross model evaluations.
    # In particular this is used for multi_choices prompts to ensure that the same
    # answer order is used for all models.
    prompt_cache: dict = Field(default_factory=dict)

    # question source and extra media
    source: QuestionSource = Field(default=None)
    medias: List[Media] = Field(default_factory=list)

    # can't use model_ : it is reserved for pydantic
    # lm_answers[prompt_version][model_version, LMAnswer]
    lm_answers: Dict[str, Dict[str, LMAnswer]] = Field(default_factory=dict)


    def add_media(self, path: str| Path, filetype: FileType = FileType.auto,
                  modality: Modality = Modality.auto):
        """add an media to the question, it is saved to the media folder
        of the benchmark archive"""
        # check if it exists

        path = Path(path)
        assert path.exists(), f"file {path} does not exist"

        FILETYPES = {".jpg": [FileType.jpg, Modality.image],
                     ".jpeg": [FileType.jpg, Modality.image],
                     ".png": [FileType.png, Modality.image],
                     ".mp4": [FileType.mp4, Modality.video],
                     ".py": [FileType.python, Modality.code],
                    }

        # auto detection if needed
        if filetype == FileType.auto or modality == Modality.auto:
            suffix = path.suffix.lower()
            if suffix in FILETYPES:
                filetype, modality = FILETYPES[suffix]
            else:
                raise ValueError(f"unsupported media format {path.suffix}")

        # compute hash
        hash = self._compute_file_hash(path)
        filename = f"{hash}{path.suffix}"

        media = Media(filetype=filetype,
                      modality=modality,
                      filename=filename,
                      size=path.stat().st_size,
                      original_path=str(path))
        self.medias.append(media)


    def _compute_file_hash(self, path: Path,
                           digest_size: int = 16) -> str:
        h = blake2b(digest_size=digest_size)
        h.update(path.read_bytes())
        return h.hexdigest()