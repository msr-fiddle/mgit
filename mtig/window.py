import curses
from enum import Enum
import pdb

"""
  This file contains several classes used to represent the terminal curses we used to
  create terminal panels used by mtig.
"""

"""
  Values used as first parameter of init_pair() subroutine of ncurses, used to represent
  the terminal color used by mtig.
"""


class ColorStyle(Enum):
    BASE = 1
    SELECTED = 2
    UNSELECTED = 3
    HIGHLIGHTED = 4
    RED = 5
    GREEN = 6
    WHITE = 7


class MainWindow:
    """
    Class representing the application main window for the whole terminal.
    """

    def __init__(self):
        """
        Perform the initialization of ncurses environment.
        """
        self.stdscr = curses.initscr()
        self.height = curses.LINES
        self.width = curses.COLS
        curses.curs_set(0)
        self.stdscr.keypad(True)
        curses.start_color()
        curses.init_pair(ColorStyle.BASE.value, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(
            ColorStyle.SELECTED.value, curses.COLOR_CYAN, curses.COLOR_BLACK
        )
        curses.init_pair(
            ColorStyle.UNSELECTED.value, curses.COLOR_WHITE, curses.COLOR_BLACK
        )
        curses.init_pair(
            ColorStyle.HIGHLIGHTED.value, curses.COLOR_BLACK, curses.COLOR_WHITE
        )
        curses.init_pair(ColorStyle.RED.value, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(ColorStyle.GREEN.value, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(ColorStyle.WHITE.value, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.noecho()
        curses.cbreak()


class InputWindow(MainWindow):
    """
    Class for input window. It creates an window with one line where the user can input
    a string.
    """

    def __init__(self, height=0, width=0, line=0, col=0):
        self.line = line
        self.col = col
        self.height = height
        self.width = width
        self.title = ""

    def getTitle(self):
        return self.title

    def setTitle(self, title):
        if len(title) < self.width - 4:
            self.title = title
        else:
            self.title = title[0 : self.width - 9] + "..."

    Title = property(getTitle, setTitle)

    def showWindow(self):
        """
        Creates a new window with the previous specified position and dimentions
        and set its title.
        """
        self.stdscr = curses.newwin(self.height, self.width, self.line, self.col)
        self.stdscr.border()
        self.stdscr.keypad(False)
        if self.title != "":
            self.stdscr.addstr(0, 2, "[%s]" % (self.title))

    def updateWindow(self):
        """
        Refresh window. It allow for example change the title and refresh with
        the new title.
        """
        self.stdscr.clear()
        self.stdscr.border()
        self.stdscr.keypad(False)
        if self.title != "":
            self.stdscr.addstr(0, 2, "[%s]" % (self.title))

    def readStr(self):
        """
        Read the user input as an string.
        """
        curses.echo()
        curses.curs_set(1)
        curses.nocbreak()
        self.stdscr.move(1, 1)
        self.stdscr.refresh()
        str = self.stdscr.getstr().decode(encoding="utf-8")
        curses.noecho()
        curses.curs_set(0)
        curses.cbreak()
        return str


class TextWindow(MainWindow):
    """
    Class representing a window with lines of text. It allows scrolling up/down/left/right.
    """

    def __init__(self, height=0, width=0, line=0, col=0):
        self.line = line
        self.col = col
        self.height = height
        self.width = width
        self.list = []
        self.colors = []
        self.num_models = 0
        self.maxWidth = 0
        self.pinNum = -1
        self.title = ""
        self.visibleBegin = 0
        self.visibleEnd = 0
        self.visibleLeft = 0
        self.visibleRight = self.visibleLeft + self.width - 4
        self.highlightSet = set()

    def readKey(self):
        key = self.stdpad.getch()
        return key

    def getTitle(self):
        return self.title

    def setTitle(self, title):
        if len(title) < self.width - 4:
            self.title = title
        else:
            self.title = title[0 : self.width - 9] + "..."

    Title = property(getTitle, setTitle)

    def getNumModels(self):
        return self.num_models

    getNumModels = property(getNumModels)

    def createLine(self, item):
        """
        Creates a line from a string (item) parameter. If item is wider than the
        window it is shrunk to its size, if it is shorter complete with blank
        spaces. Then show the characters between visibleLeft:visibleRight, in
        order to consider the left/write scrolling.
        """
        expanded = item + " " * (self.maxWidth - len(item))
        return expanded[self.visibleLeft : self.visibleRight]

    def printLine(self, line, color, idx):
        """
        Print the <line> parameter at position <idx>.
        """
        expanded = self.createLine(line)
        self.stdpad.addstr(
            idx + 1 - self.visibleBegin, 2, expanded, curses.color_pair(color.value)
        )

    def highlightItem(self, idx):
        """
        Highlight a text line alternating the background/text color, for example white
        background and black text.
        """
        if (idx >= self.visibleBegin) and (idx <= self.visibleEnd):
            self.highlightSet.add(idx)
            self.printLine(self.list[idx], ColorStyle.HIGHLIGHTED, idx)
            self.stdpad.refresh(
                0,
                0,
                self.line,
                self.col,
                self.line + self.height,
                self.col + self.width,
            )

    def highlightClear(self):
        """
        Erase any highlighting set previously.
        """
        for l in self.highlightSet:
            self.printLine(self.list[l], self.colors[l], l)
        if self.currIdx >= 0:
            self.selectItem(self.currIdx)
        self.stdpad.refresh(
            0, 0, self.line, self.col, self.line + self.height, self.col + self.width
        )
        self.highlightSet = set()

    """
    The next three methods implement the pinning of lines on the top of the text.
    """

    def setPinNum(self, n):
        self.pinNum = n

    def unsetPinNum(self):
        self.pinNum = -1

    def printPinnedLine(self, line, color, idx):
        expanded = self.createLine(line)
        self.stdpad.addstr(idx + 1, 2, expanded, curses.color_pair(color.value))

    """
    Refresh the visible text in the window.
    """

    def refreshList(self):
        if self.title != "":
            self.stdpad.addstr(0, 2, "[%s]" % (self.title))
        if self.pinNum > 0:
            idx = self.visibleBegin + self.pinNum
        else:
            idx = self.visibleBegin
        while idx < len(self.list) and (idx <= self.visibleEnd):
            if idx in self.highlightSet:
                self.printLine(self.list[idx], ColorStyle.HIGHLIGHTED, idx)
            else:
                self.printLine(self.list[idx], self.colors[idx], idx)
            idx += 1
        idx = 0
        while idx < len(self.list) and (idx < self.pinNum):
            self.printPinnedLine(self.list[idx], self.colors[idx], idx)
            idx += 1
        self.stdpad.refresh(
            0, 0, self.line, self.col, self.line + self.height, self.col + self.width
        )

    """
    Add a new line of text.
    """

    def append(self, value, color=ColorStyle.BASE):
        if len(value) > self.maxWidth:
            self.maxWidth = len(value)
        self.list.append(value)
        self.colors.append(color)
        self.num_models += 1

    def clearList(self):
        self.list = []
        self.colors = []
        self.num_models = 0
        self.pinNum = -1
        self.maxWidth = 0
        self.visibleBegin = 0
        self.visibleEnd = 0
        self.visibleLeft = 0
        self.visibleRight = self.width - 4

    def scrollDown(self, offset=1):
        self.visibleBegin += offset
        self.visibleEnd += offset
        self.refreshList()

    def scrollUp(self, offset=1):
        self.visibleBegin -= offset
        self.visibleEnd -= offset
        self.refreshList()

    """
    Define this window as the active one, therefore any command will be send to it.
    """

    def setFocus(self):
        self.stdpad.bkgd(" ", curses.color_pair(2))
        self.refreshList()

    def unsetFocus(self):
        self.stdpad.bkgd(" ", curses.color_pair(1))
        self.stdpad.refresh(
            0, 0, self.line, self.col, self.line + self.height, self.col + self.width
        )

    def resizeWidth(self, width=0):
        self.width = width
        self.visibleLeft = 0
        self.visibleRight = self.visibleLeft + self.width - 4
        self.stdpad = curses.newpad(self.height, self.width)
        self.stdpad.keypad(True)
        self.stdpad.border()
        self.refreshList()

    """
    Define this window as the active one, therefore any command will be send to it.
    """

    def updateWindow(self):
        self.stdpad.clear()
        self.stdpad.keypad(True)
        self.stdpad.border()
        self.visibleBegin = 0
        self.visibleEnd = min(
            self.num_models - 1, self.height - 3
        )  # two less because borders
        self.refreshList()

    """
    Create a new ncurse pad object and show it with the text content.
    """

    def showWindow(self):
        self.stdpad = curses.newpad(self.height, self.width)
        self.stdpad.keypad(True)
        self.stdpad.border()
        self.visibleEnd = min(
            self.num_models - 1, self.height - 3
        )  # two less because borders
        self.refreshList()

    """
    Process the subroutines associated to each key.
    """

    def keyPressed(self, key):
        if key == curses.KEY_UP:
            if self.visibleBegin > 0:
                self.scrollUp()
        elif key == curses.KEY_DOWN:
            if self.visibleEnd < self.num_models - 1:
                self.scrollDown()
        elif key == curses.KEY_PPAGE:
            page = self.visibleEnd - self.visibleBegin
            self.scrollUp(min(self.visibleBegin, page))
        elif key == curses.KEY_NPAGE:
            page = self.visibleEnd - self.visibleBegin
            self.scrollDown(min(page, (self.num_models - 1 - self.visibleEnd)))
        elif key == curses.KEY_LEFT:
            if self.visibleLeft > 0:
                self.visibleLeft -= 1
                self.visibleRight -= 1
                self.refreshList()
        elif key == curses.KEY_RIGHT:
            if self.visibleRight < self.maxWidth:
                self.visibleLeft += 1
                self.visibleRight += 1
                self.refreshList()


class BaseListWindow(TextWindow):
    """
    Class representing a window with selectable lines of text.
    """

    def __init__(self, height=0, width=0, line=0, col=0):
        self.currIdx = -1
        super().__init__(height, width, line, col)

    def getCurrentItem(self):
        if self.currIdx >= 0:
            return self.list[self.currIdx]
        else:
            return ""

    currentItem = property(getCurrentItem)

    def setPinNum(self, n):
        self.pinNum = -1  # this type of window does not support pinning

    def printPinnedLine(self, line, color, idx):
        self.printLine(line, color, idx)

    """
    Select an item, and scroll the window if needed.
    """

    def selectItem(self, idx):
        self.printLine(self.list[self.currIdx], self.colors[self.currIdx], self.currIdx)
        self.currIdx = idx
        if self.currIdx < self.visibleBegin:
            self.scrollUp(self.visibleBegin - self.currIdx)
        else:
            if self.currIdx > self.visibleEnd:
                self.scrollDown(self.currIdx - self.visibleEnd)
            else:
                self.printLine(
                    self.list[self.currIdx], ColorStyle.SELECTED, self.currIdx
                )
                self.stdpad.refresh(
                    0,
                    0,
                    self.line,
                    self.col,
                    self.line + self.height,
                    self.col + self.width,
                )

    """
    Erase selection.
    """

    def unselectItem(self):
        if self.currIdx >= 0:
            self.printLine(
                self.list[self.currIdx], self.colors[self.currIdx], self.currIdx
            )
            self.stdpad.refresh(
                0,
                0,
                self.line,
                self.col,
                self.line + self.height,
                self.col + self.width,
            )
            self.currIdx = -1

    """
    The next four subroutines does the same as TextWindow and applies selection.
    """

    def refreshList(self):
        super().refreshList()
        if self.currIdx not in self.highlightSet:
            if self.currIdx >= 0:
                self.selectItem(self.currIdx)

    def showWindow(self):
        super().showWindow()
        if self.num_models > 0:
            self.currIdx = 0
            self.selectItem(0)

    def unsetFocus(self):
        super().unsetFocus()
        if self.currIdx not in self.highlightSet:
            if self.currIdx >= 0:
                self.selectItem(self.currIdx)

    def updateWindow(self):
        if self.num_models > 0:
            self.currIdx = 0
        else:
            self.currIdx = -1
        super().updateWindow()

    """
    Process the subroutines associated to each key.
    """

    def keyPressed(self, key):
        if key == curses.KEY_UP:
            if self.currIdx > 0:
                if self.currIdx == self.visibleBegin:
                    self.scrollUp()
                self.selectItem(self.currIdx - 1)
        elif key == curses.KEY_DOWN:
            if self.currIdx < self.num_models - 1:
                if self.currIdx == self.visibleEnd:
                    self.scrollDown()
                self.selectItem(self.currIdx + 1)
        elif key == curses.KEY_PPAGE:
            page = self.visibleEnd - self.visibleBegin
            offset = min(self.visibleBegin, page)
            self.currIdx = self.visibleBegin - offset
            self.scrollUp(offset)
        elif key == curses.KEY_NPAGE:
            page = self.visibleEnd - self.visibleBegin
            offset = min(page, (self.num_models - 1 - self.visibleEnd))
            self.currIdx = self.visibleBegin + offset
            self.scrollDown(offset)

    """
    The same as TextWindow and applies selection.
    """

    def resizeWidth(self, width=0):
        super().resizeWidth(width)
        if self.num_models > 0:
            self.selectItem(self.currIdx)
        self.refreshList()
