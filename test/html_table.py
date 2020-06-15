"""
Test the generation of html code

Fred Zhang <frederic.zhang@anu.edu.au>

Australian National University
Australian Centre for Robotic Vision
"""

import pocket
import numpy as np

def test_1():

    iter1 = np.random.rand(20)
    iter2 = iter1 * 10

    a = pocket.utils.HTMLTable(4, iter1, iter2)
    a()

def test_2():

    def name_parser(name):
        seg = name.split("_")
        if seg[0] == 'p':
            return "Positive_{}<br>Score: {}<br>Prior score: {}".format(
                seg[1], seg[2], seg[3]
            )
        else:
            return "Negative_{}<br>Score: {}<br>Prior score: {}".format(
                seg[1], seg[2], seg[3]
            )

    a = pocket.utils.ImageHTMLTable(4,
            "/Users/Frederic/Desktop/example/class_161",
            parser=name_parser, width="75%")
    a()

if __name__ == '__main__':
    #test_1()
    test_2()
