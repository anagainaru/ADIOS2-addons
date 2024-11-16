from adios2 import Stream
from adios2.bindings import DerivedVarType
import numpy as np
import sys

varUName = "U"
varVName = "V"

def addDerivedToGSdata(filename):
    fOut = Stream("gs-derived.bp", "w")
    needDefine = True
    with Stream(filename, "r") as fIn:
        for _ in fIn.steps():
            varU = fIn.inquire_variable(varUName)
            varV = fIn.inquire_variable(varVName)
            U = fIn.read(varUName, start=[0]*len(varU.count()), count=varU.count())
            V = fIn.read(varVName, start=[0]*len(varV.count()), count=varV.count())

            otherVarNames = [i for i in fIn.available_variables() if i!=varUName and i!=varVName]
            otherData = {}
            for varName in otherVarNames:
                otherData[varName] = fIn.read(varName)

            if needDefine:
                outU = fOut.io.define_variable(varUName, U, U.shape, [0]*len(varU.count()), U.shape)
                outV = fOut.io.define_variable(varVName, V, V.shape, [0]*len(varV.count()), V.shape)
                dmo = fOut.io.define_derived_variable(
                        "derived/sumUV",
                        "u = "+varUName+"\nv = "+varVName+"\nu+v",
                        DerivedVarType.StatsOnly)
                needDefine = False

            fOut.begin_step()
            fOut.write(outU, U)
            fOut.write(outV, V)
            for varName in otherVarNames:
                fOut.write(varName, otherData[varName])
            fOut.end_step()

    fOut.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: %s pathToGSFile")
        exit()
    FILENAME = sys.argv[1]
    print("Reading GS file:", FILENAME)
    addDerivedToGSdata(FILENAME)
