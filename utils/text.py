from dataclasses import dataclass, field
import random
from typing import Any, Mapping


TextTemplate = list[list[str]]


@dataclass(frozen=True)
class TextGeneratorSpec:
    dataset: str
    taxonomy_prefix: tuple[str, ...] = ()
    taxonomy_fields: tuple[str, ...] = ("species",)
    template_overrides: dict[str, TextTemplate] = field(default_factory=dict)
    common_name_fallbacks: tuple[str, ...] = ("SCI", "TAX")


COMMON_TEXT_TEMPLATES: dict[str, TextTemplate] = {
    "train": [
        [
            "",
            "a photo of ",
        ],
        [
            "",
            "$AAN$ ",
        ],
        [
            "$SCI$",
            "$TAX$",
            "$COM$",
        ],
    ],
    "bioclip_sci": [["a photo of $SCI$"]],
}

DATASET_TEXT_SPECS: dict[str, TextGeneratorSpec] = {
    "bryo": TextGeneratorSpec(
        dataset="bryo",
        taxonomy_prefix=("animalia", "bryozoa", "gymnolaemata", "cheilostomatida"),
        taxonomy_fields=("family", "genus"),
    ),
    "cub": TextGeneratorSpec(
        dataset="cub",
        taxonomy_prefix=("animalia", "chordata", "aves"),
        taxonomy_fields=("order", "family", "genus", "species"),
        template_overrides={
            "train": [
                [
                    "",
                    "a photo of ",
                ],
                [
                    "",
                    "$AAN$ ",
                ],
                [
                    "$SCI$",
                    "$TAX$",
                    "$COM$",
                ],
                [
                    "",
                    " bird",
                ],
            ],
        },
    ),
    "lepid": TextGeneratorSpec(
        dataset="lepid",
        taxonomy_prefix=("animalia", "arthropoda", "insecta", "lepidoptera"),
        taxonomy_fields=("family", "genus", "species"),
        template_overrides={
            "train": [
                [
                    "",
                    "a photo of ",
                ],
                [
                    "",
                    "$AAN$ ",
                ],
                [
                    "$SCI$",
                    "$TAX$",
                    "$COM$",
                ],
                [
                    "",
                    " butterfly",
                ],
            ],
        },
    ),
    "nymph": TextGeneratorSpec(
        dataset="nymph",
        taxonomy_prefix=("animalia", "arthropoda", "insecta", "lepidoptera", "nymphalidae"),
        taxonomy_fields=("subfamily", "genus", "species"),
    ),
}


class DatasetTextGenerator:

    def __init__(self, spec: TextGeneratorSpec):
        self.spec = spec
        # Pre-build the fixed taxonomy prefix text (constant per dataset).
        self._taxonomy_prefix_text = " ".join(
            v.replace("_", " ") for v in spec.taxonomy_prefix
        )

    def get_template(self, template_name: str) -> TextTemplate:
        if template_name in self.spec.template_overrides:
            template = self.spec.template_overrides[template_name]
        elif template_name in COMMON_TEXT_TEMPLATES:
            template = COMMON_TEXT_TEMPLATES[template_name]
        else:
            raise ValueError(f"Unknown text template: '{template_name}'")

        return [list(segment_group) for segment_group in template]

    def generate(
        self,
        class_data_cid: Mapping[str, Any],
        template: TextTemplate,
        meta: Mapping[str, Any] | None = None,
    ) -> str:
        meta_data = meta or {}

        # Pre-compute all stable token values once per generate() call.
        sci_text = self._resolve_scientific_label(class_data_cid)
        tax_text = self._build_taxonomy_text(class_data_cid)
        com_text = self._resolve_common_name(class_data_cid.get("common_name"), sci_text, tax_text)
        base_tokens: dict[str, str] = {
            "SCI": sci_text,
            "TAX": tax_text,
            "COM": com_text,
            "SEX": "" if meta_data.get("sex") is None else f"{meta_data['sex']} ",
            "POS": "" if meta_data.get("pos") is None else f", {meta_data['pos']} view",
        }

        prompt = ""
        for segment_group in reversed(template):
            segment = random.choice(segment_group)
            if "$AAN$" in segment:
                segment = segment.replace("$AAN$", self._get_indefinite_article(prompt))
            rendered = segment
            for token, value in base_tokens.items():
                if f"${token}$" in rendered:
                    rendered = rendered.replace(f"${token}$", value)
            prompt = rendered + prompt

        return prompt

    def _resolve_scientific_label(self, class_data_cid: Mapping[str, Any]) -> str:
        # Prefer the most specific configured taxonomy field for this dataset.
        for field_name in reversed(self.spec.taxonomy_fields):
            value = class_data_cid.get(field_name)
            if isinstance(value, str) and value:
                return self._format_value(value)

        # Fallback for legacy/partial metadata schemas.
        for field_name in ("species", "genus", "family"):
            value = class_data_cid.get(field_name)
            if isinstance(value, str) and value:
                return self._format_value(value)

        raise ValueError(
            "Missing scientific label in class_data; expected one of "
            f"{tuple(reversed(self.spec.taxonomy_fields)) + ('species', 'genus', 'family')}"
        )

    def _resolve_common_name(self, common_name: Any, sci_text: str, tax_text: str) -> str:
        if isinstance(common_name, str) and common_name:
            return common_name

        fallback_token = random.choice(self.spec.common_name_fallbacks)
        if fallback_token == "SCI":
            return sci_text
        if fallback_token == "TAX":
            return tax_text

        raise ValueError(f"Unknown common-name fallback token: '{fallback_token}'")

    def _build_taxonomy_text(self, class_data_cid: Mapping[str, Any]) -> str:
        parts = [self._taxonomy_prefix_text] if self._taxonomy_prefix_text else []
        genus_value: str | None = None
        for field_name in self.spec.taxonomy_fields:
            value = class_data_cid.get(field_name)
            if isinstance(value, str) and value:
                if field_name == "genus":
                    genus_value = value

                if field_name == "species":
                    parts.append(self._format_species_for_taxonomy(value, genus_value))
                    continue

                parts.append(self._format_value(value))
        return " ".join(parts)

    @staticmethod
    def _format_value(value: str) -> str:
        return value.replace("_", " ")

    @staticmethod
    def _format_species_for_taxonomy(species: str, genus_value: str | None) -> str:
        if genus_value and species.startswith(f"{genus_value}_"):
            return species[len(genus_value) + 1 :].replace("_", " ")
        return species.replace("_", " ")

    @staticmethod
    def _get_indefinite_article(prompt: str) -> str:
        stripped = prompt.lstrip().lower()
        if not stripped:
            return "a"
        return "an" if stripped[0] in {"a", "e", "i", "o", "u"} else "a"

def get_text_generator(dataset: str) -> DatasetTextGenerator:
    if dataset not in DATASET_TEXT_SPECS:
        raise ValueError(f"Unknown dataset for text generation: '{dataset}'")
    return DatasetTextGenerator(DATASET_TEXT_SPECS[dataset])

def get_text_template(template_name: str, dataset: str | None = None) -> TextTemplate:
    if dataset is None:
        if template_name not in COMMON_TEXT_TEMPLATES:
            raise ValueError(f"Unknown text template: '{template_name}'")
        template = COMMON_TEXT_TEMPLATES[template_name]
        return [list(segment_group) for segment_group in template]

    return get_text_generator(dataset).get_template(template_name)