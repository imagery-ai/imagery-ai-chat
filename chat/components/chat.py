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


def action_bar_top() -> rx.Component:

    upload_box = rx.vstack(
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
            padding="2em",
            max_files=1,  # Ensure only one file can be uploaded
            on_drop=State.handle_upload(rx.upload_files(upload_id="upload1")),
        ),
        # rx.hstack(rx.foreach(rx.selected_files("upload1"), rx.text)),
        # rx.button(
        #     "Upload",
        #     on_click=State.handle_upload(rx.upload_files(upload_id="upload1")),
        # ),
        rx.foreach(State.img, lambda img: rx.image(src=rx.get_upload_url(img))),
        padding="5em",
    )

    threshold_slider = rx.vstack(
        rx.text(f"Threshold: {State.threshold}"),
        rx.slider(
            default_value=State.threshold,  # Set the default value from the state
            min=0,
            max=1,
            step=0.01,
            orientation="horizontal",
            width="300px",
            on_value_commit=State.update_threshold,  # Update the state on value commit
        ),
    )

    intensity_slider = rx.vstack(
        rx.text(f"Intensity: {State.intensity}"),
        rx.slider(
            default_value=State.intensity,  # Set the default value from the state
            min=-10,
            max=10,
            step=0.1,
            orientation="horizontal",
            width="300px",
            on_value_commit=State.update_intensity,  # Update the state on value commit)
        ),
    )

    combine = rx.center(
        rx.hstack(
            upload_box,
            rx.vstack(threshold_slider, intensity_slider, width="100%"),
            spacing="4",
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
        width="100%",
    )

    """The top action bar to send a new message with a text input and a send button."""
    return rx.vstack(
        rx.center(
            rx.hstack(
                rx.chakra.form(
                    rx.chakra.form_control(
                        rx.hstack(
                            rx.radix.text_field.root(
                                rx.radix.text_field.input(
                                    placeholder="Tell us what you want us to augment...",
                                    id="question",
                                    width=[
                                        "20em",
                                        "26em",
                                        "58.5em",
                                        "65em",
                                        "65em",
                                        "65em",
                                    ],  # Increased width by 30%
                                ),
                                rx.radix.text_field.slot(
                                    rx.tooltip(
                                        rx.icon("info", size=18),
                                        content="Enter your augmentation prompts.",
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
                    on_submit=State.process_augmentation_question,
                    reset_on_submit=True,
                ),
                spacing="4",
                align_items="stretch",
            ),
            position="sticky",
            bottom="0",
            left="0",
            padding_y="16px",
            backdrop_filter="auto",
            backdrop_blur="lg",
            border_top=f"1px solid {rx.color('mauve', 3)}",
            background_color=rx.color("mauve", 2),
            width="100%",
        ),
        combine,
    )


# def create_slider(title, value, min_value, max_value, step, width):
#     # Combining title and value into a single text component with styling
#     title_value_text = rx.text(
#         f"{title}: {value}", padding="4px 8px", border_radius="8px", text_align="center"
#     )

#     # Slider configured with given properties
#     slider = rx.slider(
#         default_value=value,
#         min=min_value,
#         max=max_value,
#         step=step,
#         orientation="horizontal",
#         width=width,
#     )

#     # Vertical stack for the title-value text and the slider
#     return rx.vstack(
#         title_value_text,  # Title-value text above the slider
#         slider,  # Actual slider below the text
#         align_items="center",  # Center-align the elements
#         width=width,
#     )


def action_bar_bottom() -> rx.Component:
    """The bottom action bar with an upload box and sliders, styled for better aesthetics."""
    upload_box = rx.vstack(
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
            padding="2em",
            max_files=1,  # Ensure only one file can be uploaded
            on_drop=State.handle_upload(rx.upload_files(upload_id="upload1")),
        ),
        # rx.hstack(rx.foreach(rx.selected_files("upload1"), rx.text)),
        # rx.button(
        #     "Upload",
        #     on_click=State.handle_upload(rx.upload_files(upload_id="upload1")),
        # ),
        rx.foreach(State.img, lambda img: rx.image(src=rx.get_upload_url(img))),
        padding="5em",
    )

    threshold_slider = rx.vstack(
        rx.text(f"Threshold: {State.threshold}"),
        rx.slider(
            default_value=State.threshold,  # Set the default value from the state
            min=0,
            max=1,
            step=0.01,
            orientation="horizontal",
            width="300px",
            on_value_commit=State.update_threshold,  # Update the state on value commit
        ),
    )

    intensity_slider = rx.vstack(
        rx.text(f"Intensity: {State.intensity}"),
        rx.slider(
            default_value=State.intensity,  # Set the default value from the state
            min=-10,
            max=10,
            step=0.1,
            orientation="horizontal",
            width="300px",
            on_value_commit=State.update_intensity,  # Update the state on value commit)
        ),
    )

    return rx.center(
        rx.hstack(
            upload_box,
            rx.vstack(threshold_slider, intensity_slider, width="100%"),
            spacing="4",
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
        width="100%",
    )


def action_bar() -> rx.Component:
    """The action bar to send a new message."""

    upload_box = rx.vstack(
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

    sliders = rx.flex(
        rx.slider(default_value=40, orientation="horizontal", height="4em"),
        rx.slider(default_value=60, orientation="horizontal", height="4em"),
        direction="column",
        spacing="4",
        width="100%",
    )

    return rx.center(
        rx.hstack(
            upload_box,
            rx.chakra.form(
                rx.chakra.form_control(
                    rx.hstack(
                        rx.radix.text_field.root(
                            rx.radix.text_field.input(
                                placeholder="Tell us what you want us to augment...",
                                id="question",
                                width=[
                                    "20em",
                                    "26em",
                                    "58.5em",
                                    "65em",
                                    "65em",
                                    "65em",
                                ],  # Increased width by 30%
                            ),
                            rx.radix.text_field.slot(
                                rx.tooltip(
                                    rx.icon("info", size=18),
                                    content="Enter your augmentation prompts.",
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
                        sliders,
                        align_items="center",
                    ),
                    is_disabled=State.processing,
                ),
                on_submit=State.process_augmentation_question,
                reset_on_submit=True,
            ),
            spacing="4",
            align_items="stretch",
        ),
        position="sticky",
        bottom="0",
        left="0",
        padding_y="16px",
        backdrop_filter="auto",
        backdrop_blur="lg",
        border_top=f"1px solid {rx.color('mauve', 3)}",
        background_color=rx.color("mauve", 2),
        width="100%",
    )

    # return rx.center(
    #     rx.vstack(
    #         rx.chakra.form(
    #             rx.chakra.form_control(
    #                 rx.hstack(
    #                     rx.radix.text_field.root(
    #                         rx.radix.text_field.input(
    #                             placeholder="Tell us what you want us to augment...",
    #                             id="question",
    #                             width=["15em", "20em", "45em", "50em", "50em", "50em"],
    #                         ),
    #                         rx.radix.text_field.slot(
    #                             rx.tooltip(
    #                                 rx.icon("info", size=18),
    #                                 content="Enter your augmentation prompts.",
    #                             )
    #                         ),
    #                     ),
    #                     rx.button(
    #                         rx.cond(
    #                             State.processing,
    #                             loading_icon(height="1em"),
    #                             rx.text("Send"),
    #                         ),
    #                         type="submit",
    #                     ),
    #                     align_items="center",
    #                 ),
    #                 is_disabled=State.processing,
    #             ),
    #             on_submit=State.process_augmentation_question,
    #             reset_on_submit=True,
    #         ),
    #     ),
    #     position="sticky",
    #     bottom="0",
    #     left="0",
    #     padding_y="16px",
    #     backdrop_filter="auto",
    #     backdrop_blur="lg",
    #     border_top=f"1px solid {rx.color('mauve', 3)}",
    #     background_color=rx.color("mauve", 2),
    #     align_items="stretch",
    #     width="100%",
    # )
