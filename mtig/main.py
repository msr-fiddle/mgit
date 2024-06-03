import os
import sys
MGIT_PATH=os.path.dirname(os.getcwd())
sys.path.append(MGIT_PATH)
from utils.lineage.graph import *  # MGit repository needs to be in PYTHONPATH.
from utils.lcs.diffcheck import diff_lcs
from utils.ht.diffcheck import diff_ht
from utils.model_utils import load_models
from utils import meta_functions
import argparse
import subprocess
import traceback
import time
from window import *
from mtig import *

"""
Read an integer provided by the user using the input window.
"""


def inputInteger(inputWindow, label):
    inputWindow.Title = label
    inputWindow.updateWindow()
    m = inputWindow.readStr()
    while not m.isdigit():
        inputWindow.updateWindow()
        m = inputWindow.readStr()
    return int(m)


def createBaseWindow(items, height, width, line, col):
    baseListWindow = BaseListWindow(height, width, line, col)
    baseListWindow.Title = "lineage"

    for i in items:
        baseListWindow.append(i)

    return baseListWindow


def main(screen):

    filename = sys.argv[1]

    if len(sys.argv) == 2:
        metadataDir = "./"
    else:
        metadataDir = sys.argv[2] + "/"

    g = LineageGraph.load_from_file("./", load_tests=True, filename=filename)

    nameMap = {}
    nameMap["root"] = "root"

    # Create the list of children models from root (DFS traversal)
    children, mapIdx = treeConnectedModels(g, "all", 0, "root")

    # Create a map between model name and the full path
    for value in children:
        shortName = extractModelNameFromPath(value[1])
        nameMap[shortName] = value[1]

    # Create the tree view composed by a list of strings
    items, modelsIdx, modelsToTreeIdx = createTreeView(children)

    # Create all the windows used by application
    mainWindow = MainWindow()

    baseListWindow = createBaseWindow(
        items, mainWindow.height - 3, mainWindow.width, 0, 0
    )

    childListWindow = TextWindow(
        mainWindow.height - 3,
        int(int(mainWindow.width + 1) / 2),
        0,
        int(mainWindow.width / 2),
    )
    childListWindow.showWindow()
    baseListWindow.showWindow()

    inputWindow = InputWindow(3, mainWindow.width, mainWindow.height - 3, 0)
    inputWindow.Title = ""
    inputWindow.showWindow()

    statusWindow = TextWindow(3, mainWindow.width, mainWindow.height - 3, 0)
    statusWindow.Title = "For help: ?"
    statusMsg = (
        "m: [m]etadata; p: [p]arenthood; s: [s]tructural diff; l: [l]ogical diff;"
    )
    statusMsg += " w: [w]rite graphical representation; j: [j]ump to a given model"
    statusWindow.append(statusMsg)
    statusWindow.showWindow()

    op = -1
    windowList = []
    windowList.append(baseListWindow)
    windowList.append(childListWindow)
    windowIdx = 0

    activeWin = windowList[windowIdx]
    activeWin.setFocus()
    numWindows = 1
    # Loop reading the commands until "q" is pressed
    while op != ord("q"):
        statusWindow.clearList()
        statusWindow.Title = "For help: ?"
        statusWindow.append(statusMsg)
        statusWindow.showWindow()
        op = activeWin.readKey()
        if op in [curses.KEY_UP, curses.KEY_DOWN, curses.KEY_PPAGE, curses.KEY_NPAGE]:
            # scroll up and down, if the right panel is present it is erased.
            if (activeWin == baseListWindow) and (childListWindow.getNumModels > 0):
                childListWindow.clearList()
                childListWindow.Title = ""
                childListWindow.updateWindow()
                baseListWindow.highlightClear()
            activeWin.keyPressed(op)
        elif op == curses.KEY_LEFT or op == curses.KEY_RIGHT:
            # scroll left and right
            activeWin.keyPressed(op)
        elif op == ord("j"):
            # Jump to a model in the left panel based on its index.
            if activeWin == baseListWindow:
                n = inputInteger(inputWindow, "Model number: ")
                m = modelsToTreeIdx[n]
                if m != baseListWindow.currIdx:
                    if childListWindow.getNumModels > 0:
                        childListWindow.clearList()
                        childListWindow.Title = ""
                        childListWindow.updateWindow()
                        baseListWindow.highlightClear()
                    # Given the model number selects it.
                    if m < baseListWindow.getNumModels:
                        baseListWindow.selectItem(m)
                    else:
                        statusWindow.clearList()
                        statusWindow.Title = "Invalid model number"
                        statusWindow.append("Invalid model number: %d" % (m))
                        statusWindow.updateWindow()
                        time.sleep(1)
        elif op == ord("w"):
            # Write a .html page with the picture of the full graph.
            g.show(layout=False, save_path=filename + ".html")
            statusWindow.clearList()
            statusWindow.Title = "Saved graphical representation"
            statusWindow.append("File name: " + filename + ".html")
            statusWindow.updateWindow()
            time.sleep(1)
        elif op == ord("p"):
            # Show the parenthood of a model in the right panel.
            if numWindows == 1:
                baseListWindow.resizeWidth(int(mainWindow.width / 2))
                activeWin.setFocus()
                numWindows = 2
            item = extractModelNameFromTree(baseListWindow.currentItem)
            lineage, mapX = treeConnectedModels(g, "all", 1, nameMap[item])
            # using the indice map of base window. The returned max are not used here (x, y)
            treeView, x, y = createTreeView(lineage, modelsIdx)
            # reset the right panel and fill with the <treeView> variable content
            childListWindow.clearList()
            baseListWindow.highlightClear()
            for value in treeView:
                childListWindow.append(value)
            childListWindow.Title = "Parenthood: " + extractModelNameFromPath(item)
            childListWindow.unsetPinNum()
            childListWindow.updateWindow()
        elif op == ord("\t"):
            # Alternate the active window (left/right).
            if numWindows == 2:
                activeWin.unsetFocus()
                windowIdx = (windowIdx + 1) % len(windowList)
                activeWin = windowList[windowIdx]
                activeWin.setFocus()
        elif op == ord("m"):
            # Load metadata file and show on right panel.
            if activeWin == baseListWindow:
                # Turn to two panels view
                if numWindows == 1:
                    baseListWindow.resizeWidth(int(mainWindow.width / 2))
                    activeWin.setFocus()
                    numWindows = 2
                childListWindow.clearList()
                baseListWindow.highlightClear()
                # Reset right panel and clean any highlight on left panel
                item = extractModelNameFromTree(baseListWindow.currentItem)
                shortName = extractModelNameFromPath(item)
                # Show just short name of the model, removing the full path
                childListWindow.Title = "Metadata: " + shortName
                metadataFileName = metadataDir + nameMap[shortName] + "/config.json"
                # Loads the metadata file
                if not os.path.isfile(metadataFileName):
                    childListWindow.append(
                        "metadata file %s not found" % (metadataFileName)
                    )
                else:
                    metadata = loadFile(metadataFileName)
                    for l in metadata:
                        childListWindow.append(l.strip())
                childListWindow.updateWindow()
                childListWindow.unsetPinNum()
        elif op == ord("s") or op == ord("l"):
            # Show the structural/logical diff on the right panel between the models
            # selected by the user.
            if activeWin == baseListWindow:
                # Turn to two panels view
                if numWindows == 1:
                    baseListWindow.resizeWidth(int(mainWindow.width / 2))
                    activeWin.setFocus()
                    numWindows = 2
                # Reset right panel and clean any highlight on left panel
                childListWindow.clearList()
                baseListWindow.highlightClear()
                if op == ord("s"):
                    childListWindow.Title = "structural diff"
                else:
                    childListWindow.Title = "logical diff"
                childListWindow.updateWindow()
                # Prompts for the two model numbers in the input window at the bottom.
                # Perform erros checking for the input (not numbers, values out of range)
                n = inputInteger(inputWindow, "Input: <# model 1>")
                m = modelsToTreeIdx[n]
                if (not m is None) and (m < baseListWindow.getNumModels):
                    m1 = children[int(m)][1]
                    m1dir = metadataDir + m1
                    # Show just short name of the model, removing the full path
                    childListWindow.append(
                        "┌ [%d] %s" % (n, extractModelNameFromPath(children[int(m)][1]))
                    )
                    childListWindow.updateWindow()
                    baseListWindow.highlightItem(m)
                    # Highlight the selected model in the left panel
                    n = inputInteger(inputWindow, "Input: <# model 2>")
                    m = modelsToTreeIdx[n]
                    if (not m is None) and (m < baseListWindow.getNumModels):
                        m2 = children[int(m)][1]
                        m2dir = metadataDir + m2
                        childListWindow.append(
                            "└ [%s] %s"
                            % (n, extractModelNameFromPath(children[int(m)][1]))
                        )
                        childListWindow.updateWindow()
                        baseListWindow.highlightItem(m)
                        mode = "contextual"
                        save_path = "output/example.html"
                        coarse = False
                        try:
                            statusWindow.clearList()
                            statusWindow.Title = "Calculating diff between models..."
                            statusWindow.append(
                                "This operation may take some time, please wait ... "
                            )
                            statusWindow.updateWindow()
                            # Models selected. Perform the diff.
                            if op == ord("s"):
                                # Structural diff using the HT method from MGit.
                                loaded_models, tracing_module_pool = load_models(
                                    [m1dir, m2dir], coarse
                                )
                                # Diff is returned on <delta> variable, instead of
                                # terminal output. The other values returned are not used
                                (
                                    add_nodes,
                                    del_nodes,
                                    add_edges,
                                    del_edges,
                                    delta,
                                ) = diff_ht(
                                    loaded_models,
                                    save_path,
                                    mode,
                                    coarse,
                                    list(tracing_module_pool),
                                    mute_terminal=True,
                                )
                                # Add diff lines using different colors depending on the operation.
                                # The operation is represented on character l[8]: "-", "+" or "*"
                                for l in delta:
                                    if l[8] == "-":
                                        childListWindow.append(l, ColorStyle.RED)
                                    elif l[8] == "+" or l[8] == "*":
                                        childListWindow.append(l, ColorStyle.GREEN)
                                    else:
                                        childListWindow.append(l, ColorStyle.BASE)
                                # Pin the two first lines of the window containing the names of the models.
                                childListWindow.setPinNum(2)
                            else:
                                # Logical diff: load the results of the tests previously done o the two models.
                                result = meta_functions.show_result_table(
                                    g, node_name_list=[m1, m2], show_metrics=True
                                )
                                for s in result.split("\n"):
                                    childListWindow.append(s)
                        except (FileNotFoundError, IOError):
                            childListWindow.append("Models data not found.")
                        childListWindow.updateWindow()
                    else:
                        childListWindow.clearList()
                        childListWindow.append("Invalid selection")
                        childListWindow.updateWindow()
                        baseListWindow.highlightClear()
                else:
                    childListWindow.clearList()
                    childListWindow.append("Invalid selection")
                    childListWindow.updateWindow()
                    baseListWindow.highlightClear()
                statusWindow.Title = "For help: ?"
                statusWindow.clearList()
                statusWindow.append(statusMsg)
                statusWindow.updateWindow()
        elif op == 27:
            # ESC: hide the right panel and makes the left panel full window.
            statusWindow.updateWindow()
            if numWindows == 2:
                baseListWindow.resizeWidth(mainWindow.width)
                activeWin = baseListWindow
                baseListWindow.setFocus()
                numWindows = 1
                baseListWindow.highlightClear()
    curses.endwin()


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            "usage: %s <lineage graph json file> <optional: metadata dir>"
            % (sys.argv[0])
        )
        sys.exit(-1)

    if not os.path.isfile(sys.argv[1]):
        print("lineage model file %s not found." % (sys.argv[1]))
        sys.exit(-1)

    if len(sys.argv) >= 3:
        if not os.path.isdir(sys.argv[2]):
            print("metadata directory %s not found." % (sys.argv[2]))
            sys.exit(-1)

    os.environ.setdefault("ESCDELAY", "1")  # ESC delay in miliseconds
    curses.wrapper(main)
