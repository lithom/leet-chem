package tech.molecules.deep.conformers;

import com.actelion.research.chem.IDCodeParser;
import com.actelion.research.chem.Molecule;
import com.actelion.research.chem.StereoIsomerEnumerator;
import com.actelion.research.chem.StereoMolecule;
import com.actelion.research.chem.conf.Conformer;
import com.actelion.research.chem.conf.ConformerSet;
import com.actelion.research.chem.conf.ConformerSetGenerator;
import com.actelion.research.chem.contrib.HydrogenHandler;
import com.actelion.research.chem.io.DWARFileParser;
import com.actelion.research.chem.prediction.MolecularPropertyHelper;
import ncsa.hdf.hdf5lib.H5;
import ncsa.hdf.hdf5lib.HDF5Constants;
import ncsa.hdf.hdf5lib.exceptions.HDF5Exception;
import ncsa.hdf.object.Datatype;
import ncsa.hdf.object.FileFormat;
import ncsa.hdf.object.Group;
import ncsa.hdf.object.h5.H5Datatype;
import ncsa.hdf.object.h5.H5File;
import org.openmolecules.chem.conf.gen.ConformerGenerator;
import tech.molecules.leet.chem.ChemUtils;

import java.io.IOException;
import java.net.URI;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.*;

public class ConformerSampler {

    private int maxAtoms = 36;
    private double cubeLengthAngstrom = 10;
    private int gridSize = 32;

    private double variances[] = new double[]{6, 2, 0.75};
    private double relOffset = 0.4;

    public static void main(String args[]) {

        DWARFileParser in = new DWARFileParser("C:\\data\\ActelionFragmentLibrary_smallFragments.dwar");

        int spx_structure = in.getSpecialFieldIndex("Structure");
        List<String> inputs = new ArrayList<>();
        while(in.next()) {
            inputs.add(in.getSpecialFieldData(spx_structure));
            if(inputs.size()>100){break;}
        }

        int batchsize = 2;
        int cnt = 0;
        for(int zi=0;zi<inputs.size();zi+=batchsize) {
            List<SpaceSampler.SampledSpaceOneHot> spaces = new ArrayList<>();
            for (String idc_i : inputs.subList( zi, Math.min(inputs.size(),zi+batchsize))) {
                try {
                    ConformerSampler cs = new ConformerSampler();
                    List<SpaceSampler.SampledSpaceOneHot> sampled = cs.sample(idc_i, 16);
                    spaces.addAll(sampled);
                    System.gc();
                    Runtime.getRuntime().gc();
                }
                catch(Exception ex) {
                    ex.printStackTrace();
                }
            }
            //writeSampledSpaceListToHDF5_2(sampled, "sampledSpaces_A.hdf5");
            try {
                NpyExporter.exportSamplesSpacesBatch(spaces, "C:\\temp\\deepspace_a", "batch_a_"+(cnt++));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
        System.out.println("mkay");
    }

    public List<SpaceSampler.SampledSpaceOneHot> sample(String idc, int maxNumPerConformerSet) {
        Random ri = new Random();
        List<SpaceSampler.SampledSpaceOneHot> samples = new ArrayList<>();

        Map<String, ConformerSet> confis = sampleConformersFromStereoIsomers(idc);
        IDCodeParser icp = new IDCodeParser();
        for (String ki : confis.keySet()) {
            for (Conformer ci : confis.get(ki).getSubset(maxNumPerConformerSet)) {

                for (int zi = 0; zi < variances.length - 1; zi++) {
                    double v1 = variances[zi] * (1.0 + 0.2 * ri.nextGaussian());
                    double v2 = variances[zi + 1] * (1.0 + 0.2 * ri.nextGaussian());

                    try {
                        StereoMolecule xi = ci.toMolecule();
                        HydrogenHandler.addImplicitHydrogens(xi);
                        xi.ensureHelperArrays(Molecule.cHelperCIP);

                        SpaceSampler.SampledSpaceOneHot sampled = sampleMolecule(xi, v1, v2, relOffset);
                        samples.add(sampled);
                    } catch (Exception ex) {
                        ex.printStackTrace();
                    }
                }
            }
        }

        return samples;
    }


    public static Map<String, ConformerSet> sampleConformersFromStereoIsomers(String idc) {
        Map<String, ConformerSet> results = new HashMap<>();
        StereoMolecule mi = ChemUtils.parseIDCode(idc);

        StereoIsomerEnumerator ie = new StereoIsomerEnumerator(mi, false);

        for (int zi = 0; zi < ie.getStereoIsomerCount(); zi++) {
            StereoMolecule si = ie.getStereoIsomer(zi);
            ConformerGenerator.addHydrogenAtoms(si);

            ConformerSetGenerator csg = new ConformerSetGenerator(200);
            ConformerSet cs = csg.generateConformerSet(si);
            results.put(si.getIDCode(), cs);
        }
        return results;
    }


    /**
     * Before and after
     *
     * @param mi
     * @param
     * @return
     */
    public SpaceSampler.SampledSpaceOneHot sampleMolecule(StereoMolecule mi, double varianceA, double varianceB, double offsetA) throws Exception {
        SpaceSampler ss = new SpaceSampler(mi, maxAtoms, cubeLengthAngstrom, gridSize, varianceA, varianceB, offsetA);
        return ss.sampleSpace(new Random());
    }

    public static void writeSampledSpaceListToHDF5_2(List<SpaceSampler.SampledSpaceOneHot2> spaces, String filename) {
        try {
            int fileId = H5.H5Fcreate(filename, HDF5Constants.H5F_ACC_TRUNC, HDF5Constants.H5P_DEFAULT, HDF5Constants.H5P_DEFAULT);
            // Create a group to store your objects
            int groupId = H5.H5Gcreate(fileId, "spaces", HDF5Constants.H5P_DEFAULT);

            // Define the compound datatype to represent your object structure
            int floatSize = 8; // Size of a double in bytes (assumed 64-bit here)
            int compoundSize = 2 * floatSize + 4; // Two doubles and an integer
            int compoundType = H5.H5Tcreate(HDF5Constants.H5T_COMPOUND, compoundSize);
            H5.H5Tinsert(compoundType, "doubleField1", 0, HDF5Constants.H5T_NATIVE_DOUBLE);
            H5.H5Tinsert(compoundType, "doubleField2", floatSize, HDF5Constants.H5T_NATIVE_DOUBLE);
            H5.H5Tinsert(compoundType, "intField", 2 * floatSize, HDF5Constants.H5T_NATIVE_INT);

            // Create a dataset for each object and write data to it
            for (int i = 0; i < spaces.size(); i++) {
                String datasetName = "object_" + i;
                int dataspaceId = H5.H5Screate_simple(1, new long[]{1}, null);
                int datasetId = H5.H5Dcreate(groupId, datasetName, compoundType, dataspaceId, HDF5Constants.H5P_DEFAULT);
                        //HDF5Constants.H5P_DEFAULT, HDF5Constants.H5P_DEFAULT, HDF5Constants.H5P_DEFAULT);
                // Extract data from your MyObject instance
                double[] doubleArray1 = spaces.get(i).x[0][0][0];
                double[] doubleArray2 = spaces.get(i).x[0][0][0];
                int[] intArray = spaces.get(i).bondsType[0][0];
                // Combine the arrays into a single array to store in the dataset
                byte[] data = new byte[compoundSize];
                ByteBuffer buffer = ByteBuffer.wrap(data);
                buffer.putDouble(doubleArray1[0]);
                buffer.putDouble(doubleArray2[0]);
                buffer.putInt(intArray[0]);
                H5.H5Dwrite(datasetId, compoundType, HDF5Constants.H5S_ALL, HDF5Constants.H5S_ALL,
                        HDF5Constants.H5P_DEFAULT, data);
                H5.H5Dclose(datasetId);
                H5.H5Sclose(dataspaceId);
            }

            // Close the group and file
            H5.H5Gclose(groupId);
            H5.H5Fclose(fileId);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }




    public static void writeSampledSpaceListToHDF5(List<SpaceSampler.SampledSpaceOneHot2> spaces, String filename) {
        // Create a new file with a given file name.

        H5File file = new H5File(filename, FileFormat.FILE_CREATE_DELETE);
        try {
            file.createFile(filename, FileFormat.FILE_CREATE_DELETE);
        } catch (Exception e) {
            e.printStackTrace();
        }

        // Open the file and write the data
        try {
            file.open();

            int index = 0;
            for (SpaceSampler.SampledSpaceOneHot2 space : spaces) {

                Group root = (Group)((javax.swing.tree.DefaultMutableTreeNode)file.getRootNode()).getUserObject();
                root.open();

                Group group = file.createGroup("SampledSpace" + index++, root);
                //Group group = root;

                // Define the dimensions of the dataset.
                long[] dims = new long[] {space.x.length, space.x[0].length, space.x[0][0].length, space.x[0][0][0].length};

                // Create and write the datasets for each field in the SampledSpace object.
                Datatype dtypeDouble = new H5Datatype(Datatype.CLASS_FLOAT, 8, Datatype.NATIVE, Datatype.SIGN_NONE);
                Datatype dtypeInt = new H5Datatype(Datatype.CLASS_INTEGER, 4, Datatype.NATIVE, Datatype.SIGN_NONE);

                file.createScalarDS("x", group, dtypeDouble, dims, null, null, 0, space.x);
                file.createScalarDS("x_target", group, dtypeDouble, dims, null, null, 0, space.x_target);

                dims = new long[] {space.structureInfo.length, space.structureInfo[0].length};
                file.createScalarDS("structureInfo", group, dtypeInt, dims, null, null, 0, space.structureInfo);

                dims = new long[] {space.bondsType.length, space.bondsType[0].length, space.bondsType[0][0].length};
                file.createScalarDS("bondsType", group, dtypeInt, dims, null, null, 0, space.bondsType);
            }

            file.close();
        } catch (HDF5Exception e) {
            System.err.println("Error writing file: " + e);
        } catch (Exception e) {
            System.err.println("Error: " + e);
        }
    }


}
