"""PPT manifest presentation tool — presents manifest to user and blocks until confirmed."""

from langchain.tools import tool


@tool("present_manifest", parse_docstring=True, return_direct=True)
def present_manifest_tool(
    manifest: str,
    slide_count: int,
) -> str:
    """Present a PPT manifest to the user for review before generating the final PPTX.

    The execution will pause and the manifest will be shown in an interactive editor.
    The user can confirm, request changes, or reject.

    Use this AFTER generating a manifest with generate_pptx_manifest and BEFORE
    submitting it to the PPTX generation service.

    IMPORTANT: Pass the manifest as a JSON string (not a Python dict).

    Args:
        manifest: The PPT manifest as a JSON string with title, slides, and theme.
        slide_count: Number of slides in the manifest.

    Returns:
        The user's response (confirmation, edit instructions, or rejection).
    """
    return "Manifest presented to user"
