""" This module contains a collection of routines to produce pretty images
"""
from __future__ import annotations

import io
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw, ImageFont

from rdkit import Chem
from rdkit.Chem import Draw

if TYPE_CHECKING:
    # pylint: disable=ungrouped-imports
    from typing import (
        Any,
        Optional,
        Dict,
        List,
        Sequence,
        Tuple,
        Union,
    )

    PilColor = Union[str, Tuple[int, int, int]]
    FrameColors = Optional[Dict[bool, PilColor]]
    from PIL.Image import Image as PilImage



def crop_image(img: PilImage, margin: int = 20) -> PilImage:
    """
    Crop an image by removing white space around it

    :param img: the image to crop
    :param margin: padding, defaults to 20
    :return: the cropped image
    """
    # pylint: disable=invalid-name
    # First find the boundaries of the white area
    x0_lim = img.width
    y0_lim = img.height
    x1_lim = 0
    y1_lim = 0
    for x in range(0, img.width):
        for y in range(0, img.height):
            if img.getpixel((x, y)) != (255, 255, 255):
                if x < x0_lim:
                    x0_lim = x
                if x > x1_lim:
                    x1_lim = x
                if y < y0_lim:
                    y0_lim = y
                if y > y1_lim:
                    y1_lim = y
    x0_lim = max(x0_lim, 0)
    y0_lim = max(y0_lim, 0)
    x1_lim = min(x1_lim + 1, img.width)
    y1_lim = min(y1_lim + 1, img.height)
    # Then crop to this area
    cropped = img.crop((x0_lim, y0_lim, x1_lim, y1_lim))
    # Then create a new image with the desired padding
    out = Image.new(
        img.mode,
        (cropped.width + 2 * margin, cropped.height + 2 * margin),
        color="white",
    )
    out.paste(cropped, (margin + 1, margin + 1))
    return out


def draw_rounded_rectangle(
    img: PilImage, color: PilColor, arc_size: int = 20
) -> PilImage:
    """
    Draw a rounded rectangle around an image

    :param img: the image to draw upon
    :param color: the color of the rectangle
    :param arc_size: the size of the corner, defaults to 20
    :return: the new image
    """
    # pylint: disable=invalid-name
    x0, y0, x1, y1 = img.getbbox()
    x1 -= 1
    y1 -= 1
    copy = img.copy()
    draw = ImageDraw.Draw(copy)
    arc_size_half = arc_size // 2
    draw.arc((x0, y0, arc_size, arc_size), start=180, end=270, fill=color)
    draw.arc((x1 - arc_size, y0, x1, arc_size), start=270, end=0, fill=color)
    draw.arc((x1 - arc_size, y1 - arc_size, x1, y1), start=0, end=90, fill=color)
    draw.arc((x0, y1 - arc_size, arc_size, y1), start=90, end=180, fill=color)
    draw.line((x0 + arc_size_half, y0, x1 - arc_size_half, y0), fill=color)
    draw.line((x1, arc_size_half, x1, y1 - arc_size_half), fill=color)
    draw.line((arc_size_half, y1, x1 - arc_size_half, y1), fill=color)
    draw.line((x0, arc_size_half, x0, y1 - arc_size_half), fill=color)
    return copy


def molecule_to_image(
    mol: Chem.rdchem.Mol, frame_color: PilColor, size: int = 300, text: str = ""
) -> PilImage:
    img = Draw.MolToImage(mol, size=(size, size))
    cropped_img = crop_image(img)
    img = draw_rounded_rectangle(cropped_img, frame_color)

    if len(text)>0:
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("Arial.ttf", 12)
        except IOError:
            font = ImageFont.load_default()
        text_width, text_height = draw.textsize(text, font=font)
        text_position = ((img.width - text_width) // 2, img.height - text_height - 10)  # Bottom-center text
        draw.text(text_position, text, fill="black", font=font)

    return img

def molecules_to_images(
    mols: Sequence[Dict[str, Any]],
    frame_colors: Sequence[PilColor],
    size: int = 300,
) -> List[PilImage]:
    all_mols = Draw.MolsToGridImage(
        [mol_data['mol'] for mol_data in mols],
        molsPerRow=len(mols),
        subImgSize=(size, size),
    )
    if not hasattr(all_mols, "crop"):  # Is not a PIL image
        fileobj = io.BytesIO(all_mols.data)
        all_mols = Image.open(fileobj)

    images = []
    for idx, mol_data in enumerate(mols):
        frame_color = frame_colors[idx]
        text = mol_data['text']
        image_obj = all_mols.crop((size * idx, 0, size * (idx + 1), size))
        image_obj = crop_image(image_obj)
        image_obj = draw_rounded_rectangle(image_obj, frame_color)
        
        if len(text)>0:
            draw = ImageDraw.Draw(image_obj)
            try:
                font = ImageFont.truetype("arial.ttf", 18)
            except IOError:
                font = ImageFont.load_default()
            (x1,y1,x2,y2) = font.getbbox(text)
            text_width, text_height = (x2-x1, y2-y1)
            text_position = ((image_obj.width - text_width) // 2, image_obj.height - text_height-4)  # Bottom-center text
            draw.text(text_position, text, fill="red", font=font)
        
        images.append(image_obj)

    return images

class RouteImageFactory:
    """
    Factory class for drawing a route

    :param route: the dictionary representation of the route
    :param in_stock_colors: the colors around molecules, defaults to {True: "green", False: "orange"}
    :param show_all: if True, also show nodes that are marked as hidden
    :param margin: the margin between images
    """

    def __init__(
        self,
        route: Dict[str, Any],
        in_stock_colors: FrameColors = None,
        show_all: bool = True,
        margin: int = 100,
    ) -> None:
        in_stock_colors = in_stock_colors or {
            True: "green",
            False: "orange",
        }
        self.show_all: bool = show_all
        self.margin: int = margin

        self._stock_lookup: Dict[str, Any] = {}
        self._mol_lookup: List[Dict[str, Any]] = []
        self._extract_molecules(route)
        images = molecules_to_images(
            self._mol_lookup,
            [in_stock_colors[mol_data['in_stock']] for mol_data in self._mol_lookup]
        )
        self._image_lookup = {mol_data['smiles']: img for mol_data, img in zip(self._mol_lookup, images)}

        self._mol_tree = self._extract_mol_tree(route)
        self._add_effective_size(self._mol_tree)

        pos0 = (
            self._mol_tree["eff_width"] - self._mol_tree["image"].width + self.margin,
            int(self._mol_tree["eff_height"] * 0.5)
            - int(self._mol_tree["image"].height * 0.5),
        )
        self._add_pos(self._mol_tree, pos0)

        self.image = Image.new(
            self._mol_tree["image"].mode,
            (self._mol_tree["eff_width"] + self.margin, self._mol_tree["eff_height"]),
            color="white",
        )
        self._draw = ImageDraw.Draw(self.image)
        self._make_image(self._mol_tree)
        self.image = crop_image(self.image)

    def _add_effective_size(self, tree_dict: Dict[str, Any]) -> None:
        children = tree_dict.get("children", [])
        for child in children:
            self._add_effective_size(child)
        if children:
            tree_dict["eff_height"] = sum(
                child["eff_height"] for child in children
            ) + self.margin * (len(children) - 1)
            tree_dict["eff_width"] = (
                max(child["eff_width"] for child in children)
                + tree_dict["image"].size[0]
                + self.margin
            )
        else:
            tree_dict["eff_height"] = tree_dict["image"].size[1]
            tree_dict["eff_width"] = tree_dict["image"].size[0] + self.margin

    def _add_pos(self, tree_dict: Dict[str, Any], pos: Tuple[int, int]) -> None:
        tree_dict["left"] = pos[0]
        tree_dict["top"] = pos[1]
        children = tree_dict.get("children")
        if not children:
            return

        mid_y = pos[1] + int(
            tree_dict["image"].height * 0.5
        )  # Mid-point of image along y
        children_height = sum(
            child["eff_height"] for child in children
        ) + self.margin * (len(children) - 1)
        childen_leftmost = min(
            pos[0] - self.margin - child["image"].width for child in children
        )
        child_y = mid_y - int(children_height * 0.5)  # Top-most edge of children
        child_ys = []
        # Now compute first guess of y-pos for children
        for child in children:
            y_adjust = int((child["eff_height"] - child["image"].height) * 0.5)
            child_ys.append(child_y + y_adjust)
            child_y += self.margin + child["eff_height"]

        for idx, (child, child_y0) in enumerate(zip(children, child_ys)):
            child_x = childen_leftmost  # pos[0] - self.margin - child["image"].width
            child_y = child_y0
            # Overwrite first guess if child does not have any children
            if not child.get("children") and idx == 0 and len(children) > 1:
                child_y = child_ys[idx + 1] - self.margin - child["image"].height
            elif not child.get("children") and idx > 0:
                child_y = (
                    child_ys[idx - 1] + self.margin + children[idx - 1]["image"].height
                )
            self._add_pos(child, (child_x, child_y))

    def _extract_mol_tree(self, tree_dict: Dict[str, Any]) -> Dict[str, Any]:
        dict_ = {
            "smiles": tree_dict["smiles"],
            "image": self._image_lookup[tree_dict["smiles"]],
        }
        if tree_dict.get("children"):
            dict_["children"] = [
                self._extract_mol_tree(grandchild)
                for grandchild in tree_dict.get("children")[0]["children"]  # type: ignore
                if not (grandchild.get("hide", False) and not self.show_all)
            ]
        return dict_

    def _extract_molecules(self, tree_dict: Dict[str, Any]) -> None:
        if tree_dict["type"] == "mol":
            self._mol_lookup.append({
                "smiles": tree_dict["smiles"],
                "mol": Chem.MolFromSmiles(tree_dict["smiles"]),
                "text": tree_dict.get("text", ""),
                "in_stock": tree_dict.get("in_stock", False)
            })
        for child in tree_dict.get("children", []):
            self._extract_molecules(child)

    def _make_image(self, tree_dict: Dict[str, Any]) -> None:
        self.image.paste(tree_dict["image"], (tree_dict["left"], tree_dict["top"]))
        children = tree_dict.get("children")
        if not children:
            return

        children_right = max(child["left"] + child["image"].width for child in children)
        mid_x = children_right + int(0.5 * (tree_dict["left"] - children_right))
        mid_y = tree_dict["top"] + int(tree_dict["image"].height * 0.5)

        self._draw.line((tree_dict["left"], mid_y, mid_x, mid_y), fill="black")
        for child in children:
            self._make_image(child)
            child_mid_y = child["top"] + int(0.5 * child["image"].height)
            self._draw.line(
                (
                    mid_x,
                    mid_y,
                    mid_x,
                    child_mid_y,
                    child["left"] + child["image"].width,
                    child_mid_y,
                ),
                fill="black",
            )
        self._draw.ellipse(
            (mid_x - 8, mid_y - 8, mid_x + 8, mid_y + 8), fill="black", outline="black"
        )
