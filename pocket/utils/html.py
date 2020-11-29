"""
Generation of html code for visualisation purposes

Fred Zhang <frederic.zhang@anu.edu.au>

Australian National University
Australian Centre for Robotic Vision
"""

import os
import numpy as np

from typing import Optional, Iterable, Callable

class HTMLTable:
    def __init__(self, num_cols: int, *args: Iterable) -> None:
        """Base class for generation of HTML tables
        Extended from code by Stephen Gould <stephen.gould@anu.edu.au>

        Arguments:
            num_cols(int): Number of columns in the table
            args(tuple of iterables): Content of the table

        Preview:
            iter_0[0], iter_0[1], ..., iter_0[M]
            iter_1[0], iter_1[1], ..., iter_1[M]
            ...
            iter_N[0], iter_N[1], ..., iter_N[M]
            iter_0[M+1], iter_0[M+2], ..., iter_0[2*M]
            ...

        Example:
            >>> import numpy as np
            >>> from pocket.utils import HTMLTable
            >>> iter1 = np.random.rand(100); iter2 = 10 * iter1
            >>> a = HTMLTable(10, iter1, iter2)
            >>> a()
        """
        self._num_cols = num_cols
        self._iterables = args

        iter_size = np.asarray([len(iterable) for iterable in args])
        if len(args) < 1:
            raise ValueError("No iterables are passed!")
        if len(np.unique(iter_size)) != 1:
            raise ValueError("All iterables passed should have the same length!")

        self._num_iter = len(args)
        self._num_rows = int(np.ceil(iter_size[0] / num_cols).item())
    
    def _page_meta(self, fp, title=None):
        """Define page meta data"""
        if title is None:
            title = "Table"
        fp.write("<head>\n")
        fp.write("<title>{}</title>\n".format(title))
        fp.write("</head>\n")

    def _table_header(self, fp):
        """Write table header"""
        pass

    def __call__(self, fname: Optional[str] = None, title: Optional[str] = None) -> None:
        """Generate html code for the table

        Arguments:
            fname(str): Name of the output html file
            title(str): Name of the html page
        """
        if fname is None:
            fname = "table.html"
        fp = open(fname, 'wt')
        fp.write("<!DOCTYPE html>\n<html>\n")
        self._page_meta(fp, title)
        fp.write("<body>\n<table>\n")
        self._table_header(fp)

        # Generate the table
        for i in range(self._num_rows):
            for j in range(self._num_iter):
                fp.write("\t<tr>\n")
                entries_each_row = self._iterables[j][i*self._num_cols: (i+1)*self._num_cols]
                for entry in entries_each_row:
                    fp.write("\t\t<td>{}</td>\n".format(entry))
                fp.write("\t</tr>\n")
        
        fp.write("</table>\n</body>\n</html>\n")
        fp.close()

class ImageHTMLTable(HTMLTable):
    def __init__(self, num_cols: int, image_dir: str,
                parser: Optional[Callable] = None,
                sorter: Optional[Callable] = None,
                extension: str = None,
                **kwargs) -> None:
        """HTML table of images with captions

        Arguments:
            num_cols(int): Number of columns in the table
            image_dir(str): Directory where images are located
            parser(callable): A parser that formats image names into captions
            sorter(callable): A function that sorts image names into desired order
            extension(str): Format of image files to be collected
            kwargs(dict): Attributes of HTML <img> tag. e.g. {"width": "75%"}
        """
        if parser is None:
            parser = lambda a: a
        if sorter is None:
            sorter = lambda a: range(len(a))
        if extension is None:
            extension = (".jpg", ".png")

        # Format attributes of <img> tag
        attr = " ".join(["{}=\"{}\"".format(k, v) for k, v in kwargs.items()])

        # Fetch image files with specified format
        files = os.listdir(image_dir); files.sort()
        all_images = [s for s in files if s.endswith(extension)]
        # Sort the images by their names
        order = sorter(all_images)
        all_image_paths = [os.path.join(image_dir, all_images[i]) for i in order]
        image_cells = ["<img src=\"{}\" ".format(im_p)+attr for im_p in all_image_paths]

        # Parse image names
        caption_cells = [parser(all_images[i]) for i in order]

        super().__init__(num_cols, image_cells, caption_cells)
