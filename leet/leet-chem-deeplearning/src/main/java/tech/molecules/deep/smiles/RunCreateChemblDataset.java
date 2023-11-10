package tech.molecules.deep.smiles;

import com.actelion.research.chem.*;

import org.nd4j.common.util.ArrayUtil;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import tech.molecules.leet.chem.ChemUtils;

import java.io.*;
import java.nio.Buffer;
import java.util.*;
import java.util.stream.Collectors;

public class RunCreateChemblDataset {

    /**
     * Length of padded input / output data
     */
    public static final int LENGTH_A = 32;

    public static final int LENGTH_B = 96;

    public static final char paddingChar = 'y';
    public static final char blindedChar = 'x';


    public static void main(String args[]) {
        //String infile_b = "C:\\Temp\\leet_input\\chembl_size26_input_smiles.csv";
        //createCSVFiles(infile_b, "b2",32);

        String infile_c = "C:\\Temp\\leet_input\\chembl_size90_input_smiles.csv";

        createCSVFiles(infile_c, "xx40", 40);
        //createCSVFiles(infile_c, "xx60", 60);
        //createCSVFiles(infile_c, "xx90_2", 90);
    }

    public static void createCSVFiles(String infile, String identifier, int length) {


        BufferedReader in = null;
        try {
            in = new BufferedReader(new FileReader(new File(infile)));
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }


        List<String> selectedMolecules = new ArrayList<>();

        try {
            String line = null;
            while ((line = in.readLine()) != null) {
                if(selectedMolecules.size() > 200000) {
                //if (selectedMolecules.size() > 200000) {
                    break;
                }
                try {
                    SmilesParser sp = new SmilesParser();
                    StereoMolecule mi = new StereoMolecule();
                    sp.parse(mi, line);
                    mi.ensureHelperArrays(Molecule.cHelperCIP);
                    int numCAtoms = ChemUtils.countAtoms(mi, Collections.singletonList(6));
                    double ratioCAtoms = (1.0 * numCAtoms) / mi.getAtoms();
                    if (ratioCAtoms < 0.4) {
                        continue;
                    }
                    if (mi.getAtoms() < 12) {
                        continue;
                    }
                    //if (mi.getAtoms() > 30) {
                    //    continue;
                    //}
                    selectedMolecules.add(mi.getIDCode());
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        // sort according to size roughly..
        Collections.shuffle(selectedMolecules);
        createCSVFiles(selectedMolecules,identifier,length);
    }
    public static void createCSVFiles(List<String> selectedMolecules, String identifier, int length) {
        System.out.println("Structures: "+selectedMolecules.size());


        // create samples blinded 0.25:
        List<TrainingSample> data_025  = createTrainingSamples(new Random(123),selectedMolecules, 0.25, 1, length);
        List<TrainingSample> data_05   = new ArrayList<>();//createTrainingSamples(new Random(124),selectedMolecules, 0.5, 1, length);
        List<TrainingSample> data_075  = new ArrayList<>();createTrainingSamples(new Random(125),selectedMolecules, 0.75, 1, length);
        List<TrainingSample> data_1    = createTrainingSamples(new Random(126),selectedMolecules, 1.0, 1, length);

        // sort the data_1 samples:
        Set<Character> padding_characters = new HashSet<>(); padding_characters.add('x'); padding_characters.add('y');
        List<TrainingSample> data_1_less_than_36    = data_1.stream().filter( xi -> xi.input_Smiles.length() - countCharacters(xi.input_Smiles,padding_characters) < 36 ).collect(Collectors.toList());
        List<TrainingSample> data_1_less_than_64    = data_1.stream().filter( xi -> xi.input_Smiles.length() - countCharacters(xi.input_Smiles,padding_characters) < 64 ).collect(Collectors.toList());


        System.out.println("okay :)");

        // find all characters to create
        Set<Character> characters_used = new HashSet<>();
        data_025.stream().map( xi -> xi.toCSV()).forEach( xi -> {for(Character ci : xi.toCharArray()) {characters_used.add(ci);}} );
        data_05.stream().map( xi -> xi.toCSV()).forEach( xi -> {for(Character ci : xi.toCharArray()) {characters_used.add(ci);}} );
        data_075.stream().map( xi -> xi.toCSV()).forEach( xi -> {for(Character ci : xi.toCharArray()) {characters_used.add(ci);}} );
        data_1.stream().map( xi -> xi.toCSV()).forEach( xi -> {for(Character ci : xi.toCharArray()) {characters_used.add(ci);}} );

        String outname_a = "smilesdata_"+identifier+"_";

        try {
            //BufferedWriter out1 = new BufferedWriter(new FileWriter(new File("smilesdata_b_025.csv")));
            BufferedWriter out1 = new BufferedWriter(new FileWriter(new File(outname_a+"025.csv")));
            for (TrainingSample ti : data_025) {
                out1.write(ti.toCSV() + "\n");
            }
            out1.flush();out1.close();
            //BufferedWriter out2 = new BufferedWriter(new FileWriter(new File("smilesdata_b_05.csv")));
            BufferedWriter out2 = new BufferedWriter(new FileWriter(new File(outname_a+"05.csv")));
            for (TrainingSample ti : data_05) {
                out2.write(ti.toCSV() + "\n");
            }
            out2.flush();
            out2.close();
            //BufferedWriter out3 = new BufferedWriter(new FileWriter(new File("smilesdata_b_075.csv")));
            BufferedWriter out3 = new BufferedWriter(new FileWriter(new File(outname_a+"075.csv")));
            for (TrainingSample ti : data_075) {
                out3.write(ti.toCSV() + "\n");
            }
            out3.flush();
            out3.close();
            //BufferedWriter out4 = new BufferedWriter(new FileWriter(new File("smilesdata_b_1.csv")));
            BufferedWriter out4 = new BufferedWriter(new FileWriter(new File(outname_a+"1_all.csv")));
            for (TrainingSample ti : data_1) {
                out4.write(ti.toCSV() + "\n");
            }
            out4.flush();
            out4.close();
            BufferedWriter out5 = new BufferedWriter(new FileWriter(new File(outname_a+"1_lessThan36.csv")));
            for (TrainingSample ti : data_1_less_than_36) {
                out5.write(ti.toCSV() + "\n");
            }
            out5.flush();
            out5.close();
            BufferedWriter out6 = new BufferedWriter(new FileWriter(new File(outname_a+"1_lessThan64.csv")));
            for (TrainingSample ti : data_1_less_than_64) {
                out6.write(ti.toCSV() + "\n");
            }
            out6.flush();
            out6.close();

        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        System.out.println("characters: "+characters_used.size());
        System.out.println("Alphabet:\n");
        StringBuilder alphabet_str = new StringBuilder();
        for(Character ci : characters_used.stream().sorted().collect(Collectors.toList())) {
            System.out.print(ci);
            alphabet_str.append(ci);
        }
        try {
            BufferedWriter out_alphabet = new BufferedWriter(new FileWriter(identifier + "_alphabet.txt"));
            out_alphabet.write(alphabet_str.toString());
            out_alphabet.flush();
            out_alphabet.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("\nmkay");
    }

    public static int countCharacters(String str, Set<Character> chars) {
        int count = 0;
        char[] ca = str.toCharArray();
        for(char ci : ca) {
            if(chars.contains(ci)) {count++;}
        }
        return count;
    }

    public static List<TrainingSample> createTrainingSamples(Random ri, List<String> mols, double blinded, int numSamplesPerMolecule, int length) {
        List<TrainingSample> samples = new ArrayList<>();

        for(int zi=0;zi<mols.size();zi++) {
            StereoMolecule mi = ChemUtils.parseIDCode(mols.get(zi));
            List<TrainingSample> samples_i = createTrainingSamples(ri,mi,blinded,numSamplesPerMolecule, length);
            samples.addAll(samples_i);
        }
        return samples;
    }

    public static List<TrainingSample> createTrainingSamples(Random ri, StereoMolecule mi, double blinded, int numSamples, int length) {

//        IDCodeParser icp = new IDCodeParser();
//        StereoMolecule mi = new StereoMolecule();
//        //icp.parse(mi,"ebQQD@@DEMDfckdPBlbbTRbTRfVRbTvbaRQ`bdUVPswtNKULtEPUEUPTDQQAH`@");
//        //icp.parse(mi,"ebQQD@@DEMDfckdPBlbbTRbTRfVRbTvbaRQ`bdUVPswtNKULtEPUEUPTDQQAH`@");
//        icp.parse(mi,"dmO@@@rdiegZFUBBbb@@");

        List<TrainingSample> samples = new ArrayList<>();

        try {
            samples = createTrainingSet(mi,ri,length,blinded,numSamples);
        } catch (Exception e) {
            e.printStackTrace();
            return samples;
        }

//        for(int zi=0;zi<samples.size();zi++) {
//            TrainingSample si = samples.get(zi);
//            System.out.println("Sample: "+si.input_Smiles+" "+si.input_CanonicBlinded+" -> "+si.output_Canonic);
//        }

        return samples;

//        // Create the 3D binary matrix using DL4J
//        int numRows = samples.size();
//        int
//        INDArray binaryMatrix = Nd4j.zeros(DataType.BOOL, numRows, numCols, numChannels);
//
//
//        // Export the binary matrix to the .npy file format
//        String filePath = "output.npy";
//        exportToCSV(binaryMatrix, filePath);

    }

    private static void exportToCSV(List<TrainingSample> samples, String filePath) {
        // Reshape the array to 1D and convert it to a byte array
        try{


        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    public static class TrainingSample {
        public final String input_Smiles;
        public final String input_CanonicBlinded;
        public final String output_Canonic;

        public TrainingSample(String input_Smiles, String input_CanonicBlinded, String output_Canonic) {
            this.input_Smiles = input_Smiles;
            this.input_CanonicBlinded = input_CanonicBlinded;
            this.output_Canonic = output_Canonic;
        }

        @Override
        public String toString() {
            return "Sample: "+this.input_Smiles+" "+this.input_CanonicBlinded+" -> "+this.output_Canonic;
        }

        public String toCSV() {
            return this.input_Smiles+","+this.input_CanonicBlinded+","+this.output_Canonic;
        }
    }



    public static List<TrainingSample> createTrainingSet(StereoMolecule mi, Random ri, int length, double blinded, int numSamples) throws Exception {
        List<TrainingSample> trainingData = new ArrayList<>();

        IsomericSmilesGenerator2 isca = new IsomericSmilesGenerator2(mi,IsomericSmilesGenerator2.MODE_INCLUDE_MAPPING);
        String canonized = isca.getSmiles();

        if(canonized.length() > length) {
            throw new Exception("Too long..");
        }

        String canonized_a = addPadding(canonized,paddingChar,length);

        for(int zi=0;zi<numSamples;zi++) {
            IsomericSmilesGenerator2 isc = new IsomericSmilesGenerator2(mi, IsomericSmilesGenerator2.MODE_INCLUDE_MAPPING,ri);
            String smi = isc.getSmiles();
            if(smi.length() > length) {continue;}
            //String smi_padded = addPadding(smi,paddingChar,length);
            double paddingDistribution = ri.nextDouble();
            String smi_padded = addPaddingBothSides(smi,paddingChar,length,paddingDistribution);

            char[] canonized_blinded = canonized_a.toCharArray();
            StringBuilder sb_blinded = new StringBuilder();
            for(int zj=0;zj<canonized.length();zj++) {
                if(ri.nextDouble()<blinded) {
                    sb_blinded.append(canonized_blinded[zj]);
                    canonized_blinded[zj] = blindedChar;
                }
            }

            //trainingData.add(new TrainingSample(smi_padded,new String(canonized_blinded),canonized_a));
            trainingData.add(new TrainingSample(smi_padded,new String(canonized_blinded),addPadding(sb_blinded.toString(),paddingChar,length)));
        }
        return trainingData;
    }

    public static String addPadding(String input, char paddingChar, int desiredLength) {
        int paddingLength = desiredLength - input.length();
        if (paddingLength <= 0) {
            // No padding needed or negative padding length
            return input;
        }

        StringBuilder paddedString = new StringBuilder(input);
        for (int i = 0; i < paddingLength; i++) {
            paddedString.append(paddingChar);
        }
        return paddedString.toString();
    }

    public static String addPaddingBothSides(String input, char paddingChar, int desiredLength, double distribution) {
        int paddingLength = desiredLength - input.length();
        if (paddingLength <= 0) {
            // No padding needed or negative padding length
            return input;
        }

        int padding_before = (int) (distribution*paddingLength);
        int padding_after  = paddingLength - padding_before;

        //StringBuilder paddedString = new StringBuilder(input);
        StringBuilder paddedString = new StringBuilder();
        for (int i = 0; i < padding_before; i++) {
            paddedString.append(paddingChar);
        }
        paddedString.append(input);
        for (int i = 0; i < padding_after; i++) {
            paddedString.append(paddingChar);
        }

        return paddedString.toString();
    }



}
