import reflex as rx

from chat.components import loading_icon
from chat.state import QA, State


message_style = dict(
    display="inline-block",
    padding="1em",
    border_radius="8px",
    max_width=["30em", "30em", "50em", "50em", "50em", "50em"],
)


def message(qa: QA) -> rx.Component:
    """A single question/answer message.

    Args:
        qa: The question/answer pair.

    Returns:
        A component displaying the question/answer pair.
    """
    return rx.cond(
        qa.image,
        rx.image(src=rx.get_upload_url(qa.image), width="100%"),
        rx.vstack(
            rx.box(
                rx.markdown(
                    qa.question,
                    background_color=rx.color("mauve", 4),
                    color=rx.color("mauve", 12),
                    **message_style,
                ),
                text_align="right",
                margin_top="1em",
            ),
            rx.box(
                rx.markdown(
                    qa.answer,
                    background_color=rx.color("accent", 4),
                    color=rx.color("accent", 12),
                    **message_style,
                ),
                text_align="left",
                padding_top="1em",
            ),
        ),
    )


def chat() -> rx.Component:
    """List all the messages in a single conversation."""
    return rx.vstack(
        rx.box(rx.foreach(State.chats[State.current_chat], message), width="100%"),
        py="8",
        flex="1",
        width="100%",
        max_width="50em",
        padding_x="4px",
        align_self="center",
        overflow="hidden",
        padding_bottom="5em",
    )


def upload_box() -> rx.Component:
    """Create an upload box for files."""
    return rx.vstack(
        rx.upload(
            rx.vstack(
                rx.button(
                    "Select Image",
                    color="rgb(107,99,246)",
                    bg="white",
                    border=f"1px solid rgb(107,99,246)",
                ),
                rx.text("Drag and drop an image here or click to select."),
            ),
            id="upload1",
            border=f"1px dotted rgb(107,99,246)",
            padding="5em",
            max_files=1,  # Ensure only one file can be uploaded
        ),
        rx.hstack(rx.foreach(rx.selected_files("upload1"), rx.text)),
        rx.button(
            "Upload",
            on_click=State.handle_upload(rx.upload_files(upload_id="upload1")),
        ),
        rx.foreach(State.img, lambda img: rx.image(src=rx.get_upload_url(img))),
        padding="5em",
    )


def action_bar() -> rx.Component:
    """The action bar to send a new message."""
    return rx.center(
        rx.vstack(
            rx.chakra.form(
                rx.chakra.form_control(
                    rx.hstack(
                        rx.radix.text_field.root(
                            rx.radix.text_field.input(
                                placeholder="Type something...",
                                id="question",
                                width=["15em", "20em", "45em", "50em", "50em", "50em"],
                            ),
                            rx.radix.text_field.slot(
                                rx.tooltip(
                                    rx.icon("info", size=18),
                                    content="Enter a question to get a response.",
                                )
                            ),
                        ),
                        rx.button(
                            rx.cond(
                                State.processing,
                                loading_icon(height="1em"),
                                rx.text("Send"),
                            ),
                            type="submit",
                        ),
                        align_items="center",
                    ),
                    is_disabled=State.processing,
                ),
                # on_submit=State.process_question,
                on_submit=State.process_augmentation_question,
                reset_on_submit=True,
            ),
            rx.text(
                "ReflexGPT may return factually incorrect or misleading responses. Use discretion.",
                text_align="center",
                font_size=".75em",
                color=rx.color("mauve", 10),
            ),
            rx.logo(margin_top="-1em", margin_bottom="-1em"),
            align_items="center",
        ),
        position="sticky",
        bottom="0",
        left="0",
        padding_y="16px",
        backdrop_filter="auto",
        backdrop_blur="lg",
        border_top=f"1px solid {rx.color('mauve', 3)}",
        background_color=rx.color("mauve", 2),
        align_items="stretch",
        width="100%",
    )
