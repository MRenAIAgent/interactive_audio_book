
import os
from typing import Optional, Type
import PyPDF2

from pydantic.v1 import BaseModel, Field

from vocode.streaming.action.base_action import BaseAction
from vocode.streaming.models.actions import ActionConfig, ActionInput, ActionOutput


class LoadBookVocodeActionConfig(ActionConfig, type="action_load_book"):  # type: ignore
    pass


class LoadBookParameters(BaseModel):
    book_name: str = Field(..., description="book name to load and read")


class LoadBookResponse(BaseModel):
    success: bool
    book_text: str = ""


class LoadBookAction(
    BaseAction[
        LoadBookVocodeActionConfig,
        LoadBookParameters,
        LoadBookResponse,
    ]
):
    description: str = """Attempts to read a book from a file.
    Input is the book name, and output is the book text.
    """
    parameters_type: Type[LoadBookParameters] = LoadBookParameters
    response_type: Type[LoadBookResponse] = LoadBookResponse

    def __init__(
        self,
        action_config: LoadBookVocodeActionConfig,
    ):
        super().__init__(
            action_config,
            quiet=True,
            should_respond="never",
            is_interruptible=False,
        )

    def _validate_book(self, book_name: str) -> bool:
        book_file = os.path.join(os.getcwd(), book_name)
        return os.path.exists(book_file)

    async def _end_of_run_hook(self) -> None:
        """This method is called at the end of the run method. It is optional but intended to be
        overridden if needed."""
        print("Successfully load book!")
    
    async def run(
        self, action_input: ActionInput[LoadBookParameters]
    ) -> ActionOutput[LoadBookResponse]:
        value = action_input.params.formatted_value

        success = self._validate_book(value)
    
        def read_pdf(pdf_path):
            """Read text from a PDF file"""
            try:
                with open(pdf_path, 'rb') as file:
                    # Create PDF reader object
                    pdf_reader = PyPDF2.PdfReader(file)
                    
                    # Extract text from all pages
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                    return text
            except Exception as e:
                print(f"Error reading PDF: {e}")
                return None
        # Read story from PDF file
        book_file = os.path.join(os.getcwd(), value)
        print("agent load book file path is ",book_file)
        pdf_text = read_pdf(book_file)
        await self._end_of_run_hook()
        return ActionOutput(
            action_type=action_input.action_config.type,
            response=LoadBookResponse(
                success=success,
                message = pdf_text,
            ),
        )
    
from typing import Dict, Sequence, Type

from vocode.streaming.action.abstract_factory import AbstractActionFactory
from vocode.streaming.action.base_action import BaseAction
from vocode.streaming.models.actions import ActionConfig

class MyCustomActionFactory(AbstractActionFactory):
    def create_action(self, action_config: ActionConfig):
        if action_config.type == "action_load_book":
            return LoadBookAction(action_config)
        else:
            raise Exception("Action type not supported by Agent config.")
