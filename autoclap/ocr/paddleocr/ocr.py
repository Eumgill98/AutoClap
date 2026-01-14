
from pydantic import model_validator
from typing import List, Any

import paddle
from paddleocr import PaddleOCR as _PaddleOCR

from autoclap.ocr.base import BaseOCR
from autoclap.core.output import OCROutput, OCRText


class PaddleOCRModel(BaseOCR):
    """PaddleOCR warpper class"""
    @model_validator(mode="after")
    def init_model(self):
        self.ocr = _PaddleOCR(
            use_angle_cls=True,
            lang=self.ln,
            enable_mkldnn=False,
        )
        print(f"OCR DEVICE : {self.device}")
        paddle.set_device(self.device)
        return self

    def predict(
        self,
        images: List[Any],
        **kwargs,
    ):
        """
        Run OCR inference.

        Args:
            images (List[Any]):
                - np.ndarray (H, W, C)
                - or image file paths

        Returns:
            Raw PaddleOCR output (list per image)
        """
        results = self.ocr.ocr(images, **kwargs)
        return results

    def structure_output(
        self, 
        outputs: List[Any]
    ) -> List[OCROutput]:
        """
        Convert PaddleOCR raw outputs to structured OCR output.

        PaddleOCR output structure:
        [
            [   # image 1
                [
                    [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
                    (text, confidence)
                ],
                ...
            ],
            [   # image 2
                ...
            ],
        ]

        Returns:
            List[OCROutput]:
                - one OCROutput per image
                - each OCROutput contains multiple OCRText entries
        """
        result = []

        for out in outputs:
            ocr_texts = []
            texts = out.get('rec_texts', [])
            scores = out.get('rec_scores', [])
                
            for text, conf in zip(texts, scores):
                ocr_texts.append(
                    OCRText(
                        text=text,
                        confidence=float(conf),
                    )
                )

            result.append(
                OCROutput(
                    texts=ocr_texts,
                )
            )

        return result


