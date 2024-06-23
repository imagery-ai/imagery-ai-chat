import json
from PIL import Image
import io
import os
import reflex as rx
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
from reflex.state import BaseState
import PIL
import requests
import torch
from io import BytesIO
from diffusers import LEditsPPPipelineStableDiffusion
from diffusers.utils import load_image
from leditspp.scheduling_dpmsolver_multistep_inject import (
    DPMSolverMultistepSchedulerInject,
)
from leditspp import StableDiffusionPipeline_LEDITS
from typing import Union
import numpy as np
from typing import Optional


load_dotenv()  # This loads the environment variables from the .env file

# Checking if the API key is set properly
if not os.getenv("OPENAI_API_KEY"):
    raise Exception("Please set OPENAI_API_KEY environment variable.")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

async_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_KEY"),
)


class QA(rx.Base):
    """A question and answer pair."""

    # question: str
    # answer: str
    question: Optional[str] = None  # Set default as None to make it optional
    answer: Optional[str] = None  # Set default as None to make it optional
    image: Optional[str] = None  # Optional field for image


DEFAULT_CHATS = {
    "Intros": [],
}


def load_image_fromurl(image: Union[str, PIL.Image.Image]):
    """
    Loads `image` to a PIL Image.

    Args:
        image (`str` or `PIL.Image.Image`):
            The image to convert to the PIL Image format.
    Returns:
        `PIL.Image.Image`:
            A PIL Image.
    """
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            image = PIL.Image.open(requests.get(image, stream=True).raw)
        elif os.path.isfile(image):
            image = PIL.Image.open(image)
        else:
            raise ValueError(
                f"Incorrect path or url, URLs must start with `http://` or `https://`, and {image} is not a valid path"
            )
    elif isinstance(image, PIL.Image.Image):
        image = image
    else:
        raise ValueError(
            "Incorrect format used for image. Should be an url linking to an image, a local path, or a PIL image."
        )
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def image_grid(imgs, rows, cols, spacing=20):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size

    grid = PIL.Image.new(
        "RGBA",
        size=(cols * w + (cols - 1) * spacing, rows * h + (rows - 1) * spacing),
        color=(255, 255, 255, 0),
    )
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i // rows * (w + spacing), i % rows * (h + spacing)))
        # print(( i // rows * w, i % rows * h))
    return grid


class ImageGenerator:
    # def __init__(self):
    #     # self.model_name = "stabilityai/stable-diffusion-3-medium-diffusers"  # Model name
    #     self.model_name = "runwayml/stable-diffusion-v1-5"  # Model name
    #     self.pipe = StableDiffusionPipeline_LEDITS.from_pretrained(
    #         self.model_name,
    #         safety_checker=None,
    #     )
    #     self.pipe.scheduler = DPMSolverMultistepSchedulerInject.from_pretrained(
    #         self.model_name,
    #         subfolder="scheduler",
    #         algorithm_type="sde-dpmsolver++",
    #         solver_order=2,
    #     )
    #     self.pipe = self.pipe.to(
    #         "mps"
    #     )  # TODO: CHANGE THIS DEPENDING ON HARDWARE (mps, cuda, intel)
    #     self.image = None  # Placeholder for your initial image

    async def generate_new_image(
        self,
        editing_prompt: list[str],
        reverse_editing_direction: list[bool],
        threshold: float,
        intensity: float,
    ):
        """
        Input: prompt (str) - Text prompt for generating the new image.
        Result: Generates a new image based on the prompt and sets it to the global image.
        Output: The new image
        """
        if self.image is None:
            raise ValueError("No initial image is set.")

        im = np.array(self.image)[:, :, :3]

        gen = torch.manual_seed(42)
        with torch.no_grad():
            _ = self.pipe.invert(
                im, num_inversion_steps=50, generator=gen, verbose=True, skip=0.15
            )
            edited_image = self.pipe(
                editing_prompt=editing_prompt,
                edit_threshold=[threshold]*len(editing_prompt),
                edit_guidance_scale=[intensity]*len(editing_prompt),
                reverse_editing_direction=reverse_editing_direction,
                use_intersect_mask=True,
            )

            print(f"threshold{threshold} | intensity{intensity}")

            # Update the global image
            self.image = edited_image.images[0]

        return self.image

    def set_initial_image(self, image_path: str = None, image_url: str = None):
        """
        Sets the initial image from a file path or a URL.
        """
        if image_path:
            self.image = PIL.Image.open(image_path).convert("RGB")
        elif image_url:
            self.image = load_image_fromurl(image_url).resize((512, 512))
        else:
            raise ValueError("Either image_path or image_url must be provided.")


class State(rx.State):
    """The app state."""

    # A dict from the chat name to the list of questions and answers.
    chats: dict[str, list[QA]] = DEFAULT_CHATS

    # The current chat name.
    current_chat = "Intros"

    # The current question.
    question: str

    # Whether we are processing the question.
    processing: bool = False

    # The name of the new chat.
    new_chat_name: str = ""

    img: list[str]


    threshold: float = 0.8  # Default threshold value
    intensity: float = 6  # Default intensity value

    def update_threshold(self, value: float):
        self.threshold = value

    def update_intensity(self, value: float):
        self.intensity = value

    def __init__(
        self,
        *args,
        parent_state: Optional[BaseState] = None,
        init_substates: bool = True,
        _reflex_internal_init: bool = False,
        **kwargs,
    ):
        super().__init__(
            *args,
            parent_state=parent_state,
            init_substates=init_substates,
            _reflex_internal_init=_reflex_internal_init,
            **kwargs,
        )
        self.image_generator = ImageGenerator()


    def create_chat(self):
        """Create a new chat."""
        # Add the new chat to the list of chats.
        self.current_chat = self.new_chat_name
        self.chats[self.new_chat_name] = []

    def delete_chat(self):
        """Delete the current chat."""
        del self.chats[self.current_chat]
        if len(self.chats) == 0:
            self.chats = DEFAULT_CHATS
        self.current_chat = list(self.chats.keys())[0]

    def set_chat(self, chat_name: str):
        """Set the name of the current chat.

        Args:
            chat_name: The name of the chat.
        """
        self.current_chat = chat_name

    @rx.var
    def chat_titles(self) -> list[str]:
        """Get the list of chat titles.

        Returns:
            The list of chat names.
        """
        return list(self.chats.keys())

    async def process_question(self, form_data: dict[str, str]):
        # Get the question from the form
        question = form_data["question"]

        self.processing = True
        qa = QA(question=question)
        self.chats[self.current_chat].append(qa)
        yield

        # Simulate processing the question with OpenAI (not implemented here)
        answer = "Simulated answer for demonstration."
        qa.answer = answer
        yield
        self.processing = False
        # # Check if the question is empty
        # if question == "":
        #     return

        # model = self.openai_process_question

        # async for value in model(question):
        #     yield value

    async def handle_upload(self, files: list[rx.UploadFile]):
        """Handle the upload of file(s).

        Args:
            files: The uploaded files.
        """
        self.processing = True  # Begin processing
        try:
            file = files[0]  # Assuming only one file is uploaded due to max_files=1
            upload_data = await file.read()
            outfile = rx.get_upload_dir() / file.filename
            with outfile.open("wb") as file_object:
                file_object.write(upload_data)

            qa_image = QA(image=file.filename)
            self.chats[self.current_chat].append(qa_image)

            # Treat the upload as a question
            question = f"Context Image: {file.filename}."
            answer = "What do you want to augment?"
            qa_question = QA(question=question, answer=answer)
            self.chats[self.current_chat].append(qa_question)
            print("handle_upload, chat:\n", self.chats)

            yield  # Allow the UI to update with the new state

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            self.processing = False  # End processing
            yield  # Optionally update the UI again to reflect the end of processing

    async def process_augmentation_question(self, form_data: dict[str, str]):
        """Process an image augmentation question.

        Args:
            form_data: The form data containing the question.
        """
        self.processing = True
        try:
            question_text = form_data["question"]
            auto_reply_answer = "Processing request..."
            qa = QA(question=question_text, answer=auto_reply_answer)
            self.chats[self.current_chat].append(qa)
            yield  # Update UI with the new question

            # Process the question through OpenAI
            response_content = await self.openai_process_question(question_text)
            json_data = json.loads(response_content)

            # Initialize the resulting lists
            editing_prompt = []
            reverse_editing_direction = []

            # Populate the lists based on "add" and "remove" fields
            for item in json_data["add"]:
                editing_prompt.append(item)
                reverse_editing_direction.append(False)

            for item in json_data["remove"]:
                editing_prompt.append(item)
                reverse_editing_direction.append(True)

            # Update the chat with the response
            self.chats[self.current_chat][
                -1
            ].answer = f"Editing prompts: {editing_prompt}, Directions: {reverse_editing_direction}"
            print("editing_prompt =", editing_prompt)
            print("reverse_editing_direction =", reverse_editing_direction)

            # Find the latest QA with an image
            latest_qa_image = None
            for qa in reversed(self.chats[self.current_chat]):
                if qa.image:
                    latest_qa_image = qa
                    break

            if latest_qa_image is None:
                print("No image found for augmentation.")
                return

            # Process the image to apply greyscale
            image_path = rx.get_upload_dir() / latest_qa_image.image

            """
            call ur function (image path)
            
            new img
            save the img file path
            

            """
            gen = self.image_generator

            gen.set_initial_image(image_path=image_path)

            augmented_img = await gen.generate_new_image(
                editing_prompt=editing_prompt,
                reverse_editing_direction=reverse_editing_direction,
                threshold=self.threshold,
                intensity=self.intensity,
            )


            # old stuff

            # with Image.open(image_path) as img:
            # greyscale_img = img.convert("L")

            # Save the greyscale image
            # new_image_name = f"greyscale_{latest_qa_image.image}"
            # new_image_path = image_path.parent / new_image_name
            # greyscale_img.save(new_image_path)
            # with Image.open(augmented_img.file) as img:

            # random 6 digit number
            num = np.random.randint(100000, 999999)

            new_image_name = f"augmented_{num}.png"
            new_image_path = image_path.parent / new_image_name
            augmented_img.save(new_image_path)

            # Append the new greyscale image to the chats
            qa_image = QA(image=new_image_name)
            self.chats[self.current_chat].append(qa_image)
            print(f"Augmented image saved and appended to chat: {new_image_path}")

        except Exception as e:
            print(f"An error occurred at process_augmentation_question: {e}")
        finally:
            self.processing = False  # End processing
            yield  # Optionally update the UI again to reflect the end of processing

    async def openai_process_question(self, question: str):
        """Fetch response from the API and return it.

        Args:
            question: str - The question text.

        Returns:
            str: The response content as a string.
        """

        system_content = "The task involves editing an image using 'add' and 'remove' prompts. Your task is to create a JSON file with two fields which is a list: 'add' and 'remove'. Based on the prompts provided, categorize the elements to be added or removed accordingly."

        response = await async_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": question},
            ],
            max_tokens=4096,
            temperature=0.5,
        )

        return response.choices[0].message.content

    async def set_image_context(self, form_data: dict[str, str]):
        """
        Input: Image
        Result: The global image is set to the image provided.
                Other functions using the global image will now use this image.
        Output: None
        """
        image_url = form_data.get("image_url")
        image_path = form_data.get("image_path")

        if image_url:
            self.image_generator.set_initial_image(image_url=image_url)
        elif image_path:
            self.image_generator.set_initial_image(image_path=image_path)
        else:
            raise ValueError("Either image_url or image_path must be provided.")

        yield

    async def generate_new_image(self, prompt: str):
        """
        Input: prompt (str) - Text prompt for generating the new image.
        Result: Based on the prompt, make alterations to the image and set the global image to the new image.
        Output: The new image
        """
        if not prompt:
            raise ValueError("Prompt cannot be empty.")

        new_image = await self.image_generator.generate_new_image(prompt)
        yield new_image
