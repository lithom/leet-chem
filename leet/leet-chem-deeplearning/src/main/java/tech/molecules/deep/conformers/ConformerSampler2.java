package tech.molecules.deep.conformers;

import com.actelion.research.chem.io.DWARFileParser;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class ConformerSampler2 {

    public static void main(String args[]) {

        DWARFileParser in = new DWARFileParser("C:\\data\\ActelionFragmentLibrary_smallFragments.dwar");

        int spx_structure = in.getSpecialFieldIndex("Structure");
        List<String> inputs = new ArrayList<>();
        while (in.next()) {
            inputs.add(in.getSpecialFieldData(spx_structure));
            if (inputs.size() > 100) {
                break;
            }
        }

        int batchsize = 2;
        int cnt = 0;
        for (int zi = 0; zi < inputs.size(); zi += batchsize) {
            List<SpaceSampler.SampledSpaceOneHot> spaces = new ArrayList<>();
            for (String idc_i : inputs.subList(zi, Math.min(inputs.size(), zi + batchsize))) {
                try {
                    ConformerSampler2 cs = new ConformerSampler2();
                    List<SpaceSampler.SampledSpaceOneHot> sampled = cs.sample(idc_i, 16);
                    spaces.addAll(sampled);
                    System.gc();
                    Runtime.getRuntime().gc();
                } catch (Exception ex) {
                    ex.printStackTrace();
                }
            }
            //writeSampledSpaceListToHDF5_2(sampled, "sampledSpaces_A.hdf5");
            try {
                NpyExporter.exportSamplesSpacesBatch(spaces, "C:\\temp\\deepspace_a", "batch_a_" + (cnt++));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
        System.out.println("mkay");
    }

    public List<SpaceSampler.SampledSpaceOneHot> sample(String idc, int maxNum) {
        return null;
    }

}
