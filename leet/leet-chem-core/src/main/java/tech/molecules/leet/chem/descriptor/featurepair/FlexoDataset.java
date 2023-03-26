package tech.molecules.leet.chem.descriptor.featurepair;

import org.apache.commons.lang3.tuple.Pair;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class FlexoDataset {

    private List<FlexoDatapoint> data;



    /**
     * NOTE: Implicit assumptions for loading / storing:
     * Empty string is mapped to 0
     * All integers from 1 to maxInteger are used to encode known strings.
     * Integer maxInteger+1 is used to encode unknown strings.
     *
     * NOTE: Currently, the mapping string <-> int is bijective. This could
     * be extended in the future.
     */
    public static class FlexoPathEncoder {
        private Map<String,Integer> encoding;
        private int intUnknown;
        public FlexoPathEncoder(Map<String, Integer> encoding, int intUnknown) {
            this.encoding = encoding;
            this.intUnknown = intUnknown;
        }

        public int encode(String pi) {
            if(pi.isEmpty()) {return 0;}
            if(encoding.containsKey(pi)) {
                return encoding.get(pi);
            }
            else {
                return intUnknown;
            }
        }

        public void store(String filename) throws IOException {
            BufferedWriter out = new BufferedWriter(new FileWriter(filename));
            for(Map.Entry<String,Integer> pi : encoding.entrySet()) {
                out.write(pi.getKey()+","+pi.getValue()+"\n");
            }
            out.flush();
            out.close();
        }

        public static FlexoPathEncoder load(String file) throws IOException {
            BufferedReader in = new BufferedReader(new FileReader(file));
            String line = null;
            int maxInt = -1;
            Map<String,Integer> map_enc = new HashMap<>();
            while( (line=in.readLine()) != null ) {
                String la[] = line.split(",");
                int ii = Integer.parseInt(la[1]);
                maxInt = Math.max(maxInt,ii);
            }
            return new FlexoPathEncoder(map_enc,maxInt+1);
        }
    }

    public static class FlexoDatapoint {
        private List<String> spheres;
        private double macrocycleScore;
        private int[] histogramCounts;

        public FlexoDatapoint(List<String> spheres, double macrocycleScore, int[] histogramCounts) {
            this.spheres = spheres;
            this.macrocycleScore = macrocycleScore;
            this.histogramCounts = histogramCounts;
        }
    }

    public FlexoDataset(List<FlexoDatapoint> data) {
        this.data = data;
    }

    public List<FlexoDatapoint> getData() {
        return this.data;
    }

    /**
     *
     * This function creates the pathsphere encoder as follows:
     * It counts the occurrences of all pathspheres and takes the
     * 254 most abundant ones, and encodes them as integers from
     * 1 to 254.
     *
     * The max path length is 32, so for each histogram, 32
     * pathspheres are encoded.
     *
     * The output is a csv file with:
     * 32 columns containing the path spheres
     * x columns that encode the histogram (x is the length of the
     *                                   provided double array)
     *
     * The histogram contains integer values corresponding to
     * the counts!
     *
     *
     * @param d
     */
    public static void createPyTorchInputData_Standard256(FlexoDataset d, String filename_Data, String filename_Encoder) throws IOException {

        // 1. count path sphere occurences:
        Map<String,Integer> counts = new HashMap<>();
        for(FlexoDatapoint pi : d.getData()) {
            if(pi.spheres.size()>32) {
                continue;
            }
            for(int zi=0;zi<pi.spheres.size();zi++) {
                String si = pi.spheres.get(zi);
                if(si.isEmpty()) {continue;}
                if(!counts.containsKey(si)) {counts.put(si,1);}
                else{counts.put(si,counts.get(si)+1);}
            }
        }

        // 2. create encoder:
        List<Pair<String,Integer>> hitcounts = counts.entrySet().stream().map( xi -> Pair.of(xi.getKey(),xi.getValue()) ).collect(Collectors.toList());
        hitcounts.sort( (x,y) -> -Integer.compare( x.getRight() , y.getRight() ) );

        // take 0-254:
        Map<String,Integer> map_encoder = new HashMap<>();
        for(int zi=0;zi<254;zi++) {
            if(hitcounts.size()>zi) {
                map_encoder.put(hitcounts.get(zi).getLeft(), zi + 1);
            }
        }

        FlexoPathEncoder encoder = new FlexoPathEncoder(map_encoder,255);
        encoder.store(filename_Encoder);

        // 3. create dataset
        createPyTorchInputData_Standard256(d,filename_Data,encoder);
    }

    public static void createPyTorchInputData_Standard256(FlexoDataset d, String filename_Data, FlexoPathEncoder encoder) throws IOException {
        // 3. create dataset
        BufferedWriter out = new BufferedWriter(new FileWriter(filename_Data));
        for(FlexoDatapoint pi : d.getData()) {
            if (pi.spheres.size() > 32) {
                continue;
            }
            // path spheres:
            List<String> strings_ps = new ArrayList<>();
            for(int zi=0;zi<32;zi++) {
                if(zi<pi.spheres.size()) {
                    strings_ps.add(""+encoder.encode(pi.spheres.get(zi)));
                }
                else {
                    strings_ps.add("0");
                }
            }
            out.write(String.join(",",strings_ps));
            //histogram
            List<String> strings_hist = new ArrayList<>();
            for(int zi=0;zi<pi.histogramCounts.length;zi++) {
                strings_hist.add(""+pi.histogramCounts[zi]);
            }
            out.write(",");
            out.write(String.join(",",strings_hist));
            out.write("\n");
        }
        out.flush();
        out.close();
    }

}
