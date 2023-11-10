package tech.molecules.deep.conformers;

import org.jetbrains.bio.npy.NpyFile;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class NpyExporter {

    public static void exportSamplesSpacesBatch(List<SpaceSampler.SampledSpaceOneHot> samples, String path, String filename_base)
            throws IOException {
        // Create the directory if it doesn't exist
        Path directoryPath = Paths.get(path);
        if (!directoryPath.toFile().exists()) {
            directoryPath.toFile().mkdirs();
        }

        // Iterate through the samples
        List<double[][][][]> batch_x         = new ArrayList<>();
        List<double[][][][]> batch_target    = new ArrayList<>();
        List<boolean[][]> batch_structureInfo    = new ArrayList<>();
        List<boolean[][][]> batch_bondsType = new ArrayList<>();

        for (int i = 0; i < samples.size(); i++) {
            // Get the data for each sample
            double[][][][] x = samples.get(i).x;
            double[][][][] target = samples.get(i).x_target;
            boolean[][] structureInfo = samples.get(i).structureInfo;
            boolean[][][] bondsType = samples.get(i).bondsType;
            batch_x.add(x);
            batch_target.add(target);
            batch_structureInfo.add(structureInfo);
            batch_bondsType.add(bondsType);
        }

        // Create file names for each .npy file
        String filenameX = filename_base + "_x.npy";
        String filenameTarget = filename_base + "_target.npy";
        String filenameStructureInfo = filename_base + "_structureInfo.npy";
        String filenameBondsType = filename_base + "_bondsType.npy";

        exportToNpy_ListX(batch_x,path,filenameX);
        exportToNpy_ListX(batch_target,path, filenameTarget);
        exportToNpy_ListAtomInfo(batch_structureInfo,path,filenameStructureInfo);
        exportToNpy_ListBondInfo(batch_bondsType,path, filenameBondsType);
    }

    public static void exportToNpy_ListX(List<double[][][][]> data, String path, String filename) {
        List<Float> unrolledDataList = new ArrayList<>();
        List<Integer> dimensions = new ArrayList<>();
        dimensions.add(data.size());
        dimensions.add(data.get(0).length);
        dimensions.add(data.get(0)[0].length);
        dimensions.add(data.get(0)[0][0].length);
        dimensions.add(data.get(0)[0][0][0].length);
        for (double[][][][] arr4D : data) {
            for (double[][][] arr3D : arr4D) {
                for (double[][] arr2D : arr3D) {
                    for (double[] arr1D : arr2D) {
                        for (double value : arr1D) {
                            unrolledDataList.add((float) value);
                        }
                    }
                }
            }
        }
        NpyFile.write(Paths.get(path,filename),convertListToArray_Float(unrolledDataList),convertListToArray_Int(dimensions));
    }

    public static void exportToNpy_ListAtomInfo(List<boolean[][]> data, String path, String filename) {
        List<Boolean> unrolledDataList = new ArrayList<>();
        List<Integer> dimensions = new ArrayList<>();
        dimensions.add(data.size());
        dimensions.add(data.get(0).length);
        dimensions.add(data.get(0)[0].length);

                for (boolean[][] arr2D : data) {
                    for (boolean[] arr1D : arr2D) {
                        for (boolean value : arr1D) {
                            unrolledDataList.add(value);
                        }
                    }
                }

        NpyFile.write(Paths.get(path,filename),convertListToArray_Bool(unrolledDataList),convertListToArray_Int(dimensions));
    }

    public static void exportToNpy_ListBondInfo(List<boolean[][][]> data, String path, String filename) {
        List<Boolean> unrolledDataList = new ArrayList<>();
        List<Integer> dimensions = new ArrayList<>();
        dimensions.add(data.size());
        dimensions.add(data.get(0).length);
        dimensions.add(data.get(0)[0].length);
        dimensions.add(data.get(0)[0][0].length);

        for(boolean[][][] arr3D : data) {
            for (boolean[][] arr2D : arr3D) {
                for (boolean[] arr1D : arr2D) {
                    for (boolean value : arr1D) {
                        unrolledDataList.add(value);
                    }
                }
            }
        }

        NpyFile.write(Paths.get(path,filename),convertListToArray_Bool(unrolledDataList),convertListToArray_Int(dimensions));
    }

    public static float[] convertListToArray_Float(List<Float> floatList) {
        // Create a new float[] array with the size of the List<Float>
        float[] floatArray = new float[floatList.size()];
        // Iterate through the List<Float> and populate the float[] array
        for (int i = 0; i < floatList.size(); i++) {
            floatArray[i] = floatList.get(i);
        }
        return floatArray;
    }

    public static int[] convertListToArray_Int(List<Integer> intList) {
        // Create a new float[] array with the size of the List<Float>
        int[] intArray = new int[intList.size()];
        // Iterate through the List<Float> and populate the float[] array
        for (int i = 0; i < intList.size(); i++) {
            intArray[i] = intList.get(i);
        }
        return intArray;
    }

    public static boolean[] convertListToArray_Bool(List<Boolean> boolList) {
        // Create a new float[] array with the size of the List<Float>
        boolean[] boolArray = new boolean[boolList.size()];
        // Iterate through the List<Float> and populate the float[] array
        for (int i = 0; i < boolList.size(); i++) {
            boolArray[i] = boolList.get(i);
        }
        return boolArray;
    }


    public static void main(String[] args) {
        // Sample 4-dimensional array
        double[][][][] data = {
                {
                        {{1.1, 1.2}, {1.3, 1.4}},
                        {{2.1, 2.2}, {2.3, 2.4}}
                },
                {
                        {{3.1, 3.2}, {3.3, 3.4}},
                        {{4.1, 4.2}, {4.3, 4.4}}
                }
        };

        // Path to the .npy file
        Path filePath = Paths.get("data.npy");

        // Write the 4-dimensional array to the .npy file
        write(filePath, data);
    }

    public static void write(Path p, double[][][][] data) {
        // Get the total size of the 4D array
        int size = getTotalSize(data);

        // Unroll the 4D array into a 1D float array
        float[] unrolledData = unroll(data, size);

        // Get the dimensions of the 4D array
        int[] dimensions = getDimensions(data);

        // Write the data to the .npy file using the provided NpyFile.write function
        NpyFile.write(p, unrolledData, dimensions);
    }

    // Helper method to calculate the total size of the 4D array
    private static int getTotalSize(double[][][][] data) {
        int size = 0;
        for (double[][][] arr3D : data) {
            for (double[][] arr2D : arr3D) {
                for (double[] arr1D : arr2D) {
                    size += arr1D.length;
                }
            }
        }
        return size;
    }

    // Helper method to unroll the 4D array into a 1D float array
    private static float[] unroll(double[][][][] data, int size) {
        float[] unrolledData = new float[size];
        int index = 0;
        for (double[][][] arr3D : data) {
            for (double[][] arr2D : arr3D) {
                for (double[] arr1D : arr2D) {
                    for (double value : arr1D) {
                        unrolledData[index] = (float) value;
                        index++;
                    }
                }
            }
        }
        return unrolledData;
    }

    // Helper method to get the dimensions of the 4D array
    private static int[] getDimensions(double[][][][] data) {
        int[] dimensions = new int[4];
        dimensions[0] = data.length;
        dimensions[1] = data[0].length;
        dimensions[2] = data[0][0].length;
        dimensions[3] = data[0][0][0].length;
        return dimensions;
    }



}
