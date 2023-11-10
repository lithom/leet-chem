package tech.molecules.deep.smiles;

import com.actelion.research.chem.*;
import com.actelion.research.chem.conf.Conformer;
import com.actelion.research.chem.conf.ConformerSet;
import com.actelion.research.chem.conf.ConformerSetGenerator;
import org.jetbrains.bio.npy.NpyFile;
import tech.molecules.leet.chem.ChemUtils;

import java.io.*;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;

public class RunSimpleChemblSmilesDataset {

    /**
     * Length of padded input / output data
     */
    public static final int LENGTH_STRING_SEQUENCE = 64;

    public static final int LENGTH_MAX_SMILES = 40;

    public static final int NUM_ATOMS = 32;

    public static final int LENGTH_ALPHABET_MAX = 50;

    public static final int DIST_MAX = 16;


    public static final int ADJ_MATRICES_AT_DIST_MAX_DISTANCE = 8;

    public static final int ATOM_TYPES_AT_DIST_MAX_DISTANCE = 4;

    /**
     * Defines the number of ones in the "problem description string" and the
     * size fo the distance matrices.
     */
    public static final int NUM_DISTANCES = 8;


    public static final char paddingChar = 'y';
    //public static final char blindedChar = 'x';


    public static void main(String args[]) {
        //String infile_b = "C:\\Temp\\leet_input\\chembl_size26_input_smiles.csv";
        //createCSVFiles(infile_b, "b2",32);

        String infile_c = "C:\\Temp\\leet_input\\chembl_size90_input_smiles.csv";

        createDistanceMatrixDataset(infile_c, "smi64_atoms32_alphabet50_MEDIUM_NEW_01");
        //createCSVFiles(infile_c, "xx60", 60);
        //createCSVFiles(infile_c, "xx90_2", 90);
    }


    public static void createDistanceMatrixDataset(String infile, String identifier) {


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
                if (selectedMolecules.size() > 40000) {
                    //if (selectedMolecules.size() > 200000) {
                    break;
                }
                try {
                    SmilesParser sp = new SmilesParser();
                    StereoMolecule mi = new StereoMolecule();
                    sp.parse(mi, line);
                    mi.stripSmallFragments();
                    mi.ensureHelperArrays(Molecule.cHelperCIP);
                    mi.stripIsotopInfo();
                    mi.removeExplicitHydrogens();
                    int numCAtoms = ChemUtils.countAtoms(mi, Collections.singletonList(6));
                    double ratioCAtoms = (1.0 * numCAtoms) / mi.getAtoms();
                    if (ratioCAtoms < 0.4) {
                        continue;
                    }
                    if (mi.getAtoms() < 8) {
                        continue;
                    }
                    //if (mi.getAtoms() > 32) {
                    if (mi.getAtoms() > 20) {
                        continue;
                    }
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
        List<SmilesMolSample> data = createDistanceMatrixDataset(selectedMolecules, true);
        //List<SmilesMolSample3D> data_3d = createConformerDataset(data,8);
        List<SmilesMolSample3D> data_fake3d = data.stream().map(xi -> new SmilesMolSample3D(xi,new ArrayList<>())).collect(Collectors.toList());
        exportSmilesSamples(data_fake3d,identifier,false);
    }


    public static List<Integer> pickRandomNumbers(int n, int x) {
        if (x > n) {
            throw new IllegalArgumentException("Cannot pick more numbers than the range allows.");
        }
        List<Integer> numbers = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            numbers.add(i);
        }
        Collections.shuffle(numbers);
        return numbers.subList(0, x);
    }

    /**
     * We only consider well defined molecules without stereo isomers..
     *
     * @param samples
     * @param maxConformersPerStereoIsomer
     * @return
     */
    public static List<SmilesMolSample3D> createConformerDataset(List<SmilesMolSample> samples, int maxConformersPerStereoIsomer) {
        List<SmilesMolSample3D> conformers3D = new ArrayList<>();
        for(SmilesMolSample xi : samples) {
            try {
                StereoMolecule mi = ChemUtils.parseIDCode(xi.idcode);
                Canonizer ca = new Canonizer(mi);
                mi = ca.getCanMolecule();

                StereoIsomerEnumerator sie = new StereoIsomerEnumerator(mi, false);
                if (sie.getStereoIsomerCount() > 1) {
                    System.out.println("Skip, too many stereo isomers: " + sie.getStereoIsomerCount());
                }

                ConformerSetGenerator csg = new ConformerSetGenerator(maxConformersPerStereoIsomer);
                ConformerSet csi = csg.generateConformerSet(mi);

                List<Coordinates[]> data3D = new ArrayList<>();
                List<double[][]> data3D_1 = new ArrayList<>();
                for (Conformer cx : csi) {
                    StereoMolecule mx = cx.toMolecule();

                    // apply random transformation?
                    if (true) {
                        Random ri = new Random();
                        double rotX = ri.nextDouble();
                        double rotY = ri.nextDouble();
                        double rotZ = ri.nextDouble();
                        for (int zi = 0; zi < mx.getAllAtoms(); zi++) {
                            double xx = mx.getAtomX(zi);
                            double yy = mx.getAtomY(zi);
                            double zz = mx.getAtomZ(zi);
                            double xr[] = rotate3D(xx, yy, zz, rotX, rotY, rotZ);
                            mx.setAtomX(zi, xr[0]);
                            mx.setAtomY(zi, xr[1]);
                            mx.setAtomZ(zi, xr[2]);
                        }
                    }

                    //Coordinates[] ci = mx.getAtomCoordinates();
                    //data3D.add(ci);

                    double[][] coords = new double[3][NUM_ATOMS];
                    //for (int zi = 0; zi < data3D.size(); zi++) {
                    for (int za = 0; za < NUM_ATOMS; za++) {
                        if(za<mx.getAtoms()) {
                            coords[0][xi.mapCanonicalToSmiles.get(za)] = mx.getAtomX(za);
                            coords[1][xi.mapCanonicalToSmiles.get(za)] = mx.getAtomY(za);
                            coords[2][xi.mapCanonicalToSmiles.get(za)] = mx.getAtomZ(za);
                        }
                    }
                    //}
                    data3D_1.add(coords);
                }
                System.out.println("Created conformers: " + data3D_1.size());

                conformers3D.add(new SmilesMolSample3D(xi, data3D_1));
            }
            catch(Exception ex) {
                ex.printStackTrace();
            }
        }
        return conformers3D;
    }

    public static double[] rotate3D(double x, double y, double z, double rotationX, double rotationY, double rotationZ) {
        // Convert rotation angles from degrees to radians
        double radRotationX = Math.toRadians(rotationX);
        double radRotationY = Math.toRadians(rotationY);
        double radRotationZ = Math.toRadians(rotationZ);

        // Apply 3D rotation formulas
        double rotatedX = x * Math.cos(radRotationY) * Math.cos(radRotationZ)
                - y * (Math.cos(radRotationX) * Math.sin(radRotationZ)
                - Math.sin(radRotationX) * Math.sin(radRotationY) * Math.cos(radRotationZ))
                + z * (Math.sin(radRotationX) * Math.sin(radRotationZ)
                + Math.cos(radRotationX) * Math.sin(radRotationY) * Math.cos(radRotationZ));

        double rotatedY = x * Math.cos(radRotationY) * Math.sin(radRotationZ)
                + y * (Math.cos(radRotationX) * Math.cos(radRotationZ)
                + Math.sin(radRotationX) * Math.sin(radRotationY) * Math.sin(radRotationZ))
                - z * (Math.sin(radRotationX) * Math.cos(radRotationZ)
                - Math.cos(radRotationX) * Math.sin(radRotationY) * Math.sin(radRotationZ));

        double rotatedZ = -x * Math.sin(radRotationY)
                + y * Math.sin(radRotationX) * Math.cos(radRotationY)
                + z * Math.cos(radRotationX) * Math.cos(radRotationY);

        return new double[]{rotatedX, rotatedY, rotatedZ};
    }

    public static class SmilesMolSample {

        public final String idcode;

        public final Map<Integer,Integer> mapCanonicalToSmiles;
        public final String smilesWithProblem;

        public final int[][] fullDistMatrix;
        public final int[][] smallDistMatrix;
        public final int[]   chemInfo;
        public final int[]   distForFirst;

        public final boolean[][][] distAdjMatrices;

        public final byte[][][] atomTypesAtDistanceN;

        public SmilesMolSample(String idcode, Map<Integer,Integer> mapCanonicalToSmiles, String smilesWithProblem,
                               int[][] fullDistMatrix, int[][] smallDistMatrix, int[] chemInfo, int[] distForFirst,
                               boolean[][][] distAdjMatrices, byte[][][] atomTypesAtDistanceN) {
            this.idcode = idcode;
            this.mapCanonicalToSmiles = mapCanonicalToSmiles;
            this.smilesWithProblem = smilesWithProblem;
            this.fullDistMatrix  = fullDistMatrix;
            this.smallDistMatrix = smallDistMatrix;
            this.chemInfo = chemInfo;
            this.distForFirst = distForFirst;
            this.distAdjMatrices = distAdjMatrices;
            this.atomTypesAtDistanceN = atomTypesAtDistanceN;
        }
    }

    public static class SmilesMolSample3D extends SmilesMolSample {
        public final List<double[][]> conformations;

        public SmilesMolSample3D(SmilesMolSample sample, List<double[][]> conformations) {
            this(sample.idcode,sample.mapCanonicalToSmiles,sample.smilesWithProblem,
                    sample.fullDistMatrix,sample.smallDistMatrix,
                    sample.chemInfo,sample.distForFirst,
                    sample.distAdjMatrices, sample.atomTypesAtDistanceN,
                    conformations);
        }

        public SmilesMolSample3D(String idcode, Map<Integer, Integer> mapCanonicalToSmiles, String smilesWithProblem,
                                 int[][] fullDistMatrix, int[][] smallDistMatrix, int[] chemInfo, int[] distForFirst,
                                 boolean[][][] distAdjMatrices, byte[][][] atomTypesAtDistanceN,
                                 List<double[][]> conformations) {
            super(idcode, mapCanonicalToSmiles, smilesWithProblem, fullDistMatrix, smallDistMatrix,
                    chemInfo, distForFirst,distAdjMatrices,atomTypesAtDistanceN);
            this.conformations = conformations;
        }
    }

    /**
     *
     * @param mols
     * @param randomOrCanonicalSmiles true: random, false: canonical
     */
    public static List<SmilesMolSample> createDistanceMatrixDataset(List<String> mols, boolean randomOrCanonicalSmiles) {

        Random ri = new Random(123);

        //List<Triple<String,int[][],int[]>> data = new ArrayList<>();
        List<SmilesMolSample> data = new ArrayList<>();

        for (int zi = 0; zi < mols.size(); zi++) {
            for (int zz = 0; zz < 4; zz++) {
                StereoMolecule mi = ChemUtils.parseIDCode(mols.get(zi));
                Canonizer ci = new Canonizer(mi);
                mi = ci.getCanMolecule();
                mi.ensureHelperArrays(Molecule.cHelperCIP);
                if (mi.getAtoms() > NUM_ATOMS) {
                    continue;
                }
                //IsomericSmilesGenerator2 isc = new IsomericSmilesGenerator2(mi, IsomericSmilesGenerator2.MODE_INCLUDE_MAPPING, ri);
                IsomericSmilesGenerator2 isc = null;
                if (randomOrCanonicalSmiles) {
                    isc = new IsomericSmilesGenerator2(mi, IsomericSmilesGenerator2.MODE_INCLUDE_MAPPING, ri);
                } else {
                    isc = new IsomericSmilesGenerator2(mi, IsomericSmilesGenerator2.MODE_INCLUDE_MAPPING);
                }
                String smi = isc.getSmiles();
                if (smi.length() > LENGTH_MAX_SMILES) {
                    System.out.println("Smiles too long: " + smi.length());
                    continue;
                }
                // pick atoms for distance matrix: (!!! MUST THEN BE SORTED OF COURSE !!!)
                List<Integer> adjMatrixAtoms = pickRandomNumbers(mi.getAtoms(), NUM_DISTANCES);
                adjMatrixAtoms = adjMatrixAtoms.stream().sorted().collect(Collectors.toList());
                StringBuilder sb_problemDescription = new StringBuilder();
                for (int xi = 0; xi < mi.getAtoms(); xi++) {
                    if (adjMatrixAtoms.contains(xi)) {
                        sb_problemDescription.append("1");
                    } else {
                        sb_problemDescription.append("0");
                    }
                }

                Map<Integer, Integer> mapCanonicalToSmiles = isc.getMapOrderCanonizedToSmiles();

                int[][] fullDistanceMap = new int[NUM_ATOMS][NUM_ATOMS];
                for (int za = 0; za < mi.getAtoms(); za++) {
                    for (int zb = 0; zb < mi.getAtoms(); zb++) {
                        int dab = mi.getPathLength(za, zb);
                        dab = Math.min(dab, DIST_MAX);
                        fullDistanceMap[mapCanonicalToSmiles.get(za)][mapCanonicalToSmiles.get(zb)] = dab;
                    }
                }
                int[][] smallDistanceMap = new int[NUM_DISTANCES][NUM_DISTANCES];
                for (int za = 0; za < NUM_DISTANCES; za++) {
                    for (int zb = 0; zb < NUM_DISTANCES; zb++) {
                        int dab = fullDistanceMap[adjMatrixAtoms.get(za)][adjMatrixAtoms.get(zb)];
                        dab = Math.min(dab, DIST_MAX);
                        smallDistanceMap[za][zb] = dab;
                    }
                }

                String smi_with_problem_description = smi + "y" + sb_problemDescription.toString();
                if (smi_with_problem_description.length() > LENGTH_STRING_SEQUENCE) {
                    System.out.println("String sequence too long: " + smi_with_problem_description.length());
                    continue;
                }

                // pad string to have correct length: pad randomly
                //StringBuilder sb_smi_padded = new StringBuilder(smi);
                int pad_before = ri.nextInt(LENGTH_STRING_SEQUENCE - smi_with_problem_description.length() + 1);
                int pad_after = LENGTH_STRING_SEQUENCE - (pad_before + smi_with_problem_description.length());
                StringBuilder sb_pad_before = new StringBuilder();
                StringBuilder sb_pad_after = new StringBuilder();
                for (int xi = 0; xi < pad_before; xi++) {
                    sb_pad_before.append(paddingChar);
                }
                for (int xi = 0; xi < pad_after; xi++) {
                    sb_pad_after.append(paddingChar);
                }
                String smi_padded = sb_pad_before.toString() + smi_with_problem_description + sb_pad_after.toString();


                int[] chemInfoStuff = new int[mi.getAtoms() * 2];
                for (int za = 0; za < mi.getAtoms(); za++) {
                    int connAtoms = mi.getConnAtoms(za);
                    int ringAtom = mi.isRingAtom(za) ? 1 : 0;
                    chemInfoStuff[2 * mapCanonicalToSmiles.get(za) + 0] = connAtoms;
                    chemInfoStuff[2 * mapCanonicalToSmiles.get(za) + 1] = ringAtom;
                }

                int[] distancesFromFirst = new int[NUM_ATOMS];
                for (int za = 0; za < mi.getAtoms(); za++) {
                    distancesFromFirst[za] = fullDistanceMap[0][za];
                }


                // Create "adjacency matrix at distance n"
                boolean[][][] distAdjMatrices = new boolean[ADJ_MATRICES_AT_DIST_MAX_DISTANCE][NUM_ATOMS][NUM_ATOMS];
                for(int di = 0; di<distAdjMatrices.length;di++) {
                    for (int za = 0; za < mi.getAtoms(); za++) {
                        Set<Integer> atomsAtDist = findAllAtomsAtExactDistance(mi,za,di+1,new boolean[NUM_ATOMS]);
                        for (int zb : atomsAtDist) {
                            distAdjMatrices[di][ mapCanonicalToSmiles.get(za) ][ mapCanonicalToSmiles.get(zb) ] = true;
                            distAdjMatrices[di][ mapCanonicalToSmiles.get(zb) ][ mapCanonicalToSmiles.get(za) ] = true;
                        }
                    }
                }

                // Create count atom types at distance n (dist,atom_root,type)
                byte[][][] atomTypesAtDistanceN = new byte[NUM_ATOMS][ATOM_TYPES_AT_DIST_MAX_DISTANCE][13];
                for(int di = 0; di<atomTypesAtDistanceN[0].length;di++) {

                    for (int za = 0; za < mi.getAtoms(); za++) {
                        byte cnt_c = 0; byte cnt_n = 0;
                        byte cnt_o = 0; byte cnt_f = 0;
                        byte cnt_s = 0; byte cnt_cl = 0;
                        byte cnt_p = 0; byte cnt_br = 0;
                        byte cnt_arom = 0;
                        byte deg_1 = 0; byte deg_2 = 0;
                        byte deg_3 = 0; byte deg_4 = 0;

                        Set<Integer> atomsAtDist = findAllAtomsAtExactDistance(mi,za,di+1,new boolean[NUM_ATOMS]);
                        for (int zb : atomsAtDist) {
                            //if(mi.getPathLength(za,zb)==di) {
                                if(mi.getAtomicNo(zb)==6){cnt_c++;}
                                if(mi.getAtomicNo(zb)==7){cnt_n++;}
                                if(mi.getAtomicNo(zb)==8){cnt_o++;}
                                if(mi.getAtomicNo(zb)==9){cnt_f++;}
                                if(mi.getAtomicNo(zb)==16){cnt_s++;}
                                if(mi.getAtomicNo(zb)==17){cnt_cl++;}
                                if(mi.getAtomicNo(zb)==15){cnt_p++;}
                                if(mi.getAtomicNo(zb)==35){cnt_br++;}
                                if(mi.isAromaticAtom(zb)){cnt_arom++;}
                                if(mi.getConnAtoms(zb)==1){deg_1++;}
                                if(mi.getConnAtoms(zb)==2){deg_2++;}
                                if(mi.getConnAtoms(zb)==3){deg_3++;}
                                if(mi.getConnAtoms(zb)==4){deg_4++;}
                            //}
                        }
                        atomTypesAtDistanceN[mapCanonicalToSmiles.get(za)][di][0] = cnt_c;
                        atomTypesAtDistanceN[mapCanonicalToSmiles.get(za)][di][1] = cnt_n;
                        atomTypesAtDistanceN[mapCanonicalToSmiles.get(za)][di][2] = cnt_o;
                        atomTypesAtDistanceN[mapCanonicalToSmiles.get(za)][di][3] = cnt_f;
                        atomTypesAtDistanceN[mapCanonicalToSmiles.get(za)][di][4] = cnt_s;
                        atomTypesAtDistanceN[mapCanonicalToSmiles.get(za)][di][5] = cnt_cl;
                        atomTypesAtDistanceN[mapCanonicalToSmiles.get(za)][di][6] = cnt_p;
                        atomTypesAtDistanceN[mapCanonicalToSmiles.get(za)][di][7] = cnt_br;
                        atomTypesAtDistanceN[mapCanonicalToSmiles.get(za)][di][8] = cnt_arom;
                        atomTypesAtDistanceN[mapCanonicalToSmiles.get(za)][di][9] = deg_1;
                        atomTypesAtDistanceN[mapCanonicalToSmiles.get(za)][di][10] = deg_2;
                        atomTypesAtDistanceN[mapCanonicalToSmiles.get(za)][di][11] = deg_3;
                        atomTypesAtDistanceN[mapCanonicalToSmiles.get(za)][di][12] = deg_4;
                    }
                }

                //data.add(Triple.of(smi_padded, smallDistanceMap, chemInfoStuff));
                data.add(new SmilesMolSample(mols.get(zi), mapCanonicalToSmiles, smi_padded, fullDistanceMap,
                        smallDistanceMap, chemInfoStuff, distancesFromFirst,
                        distAdjMatrices, atomTypesAtDistanceN
                        ));
            }
        }

        return data;
    }

    public static Set<Integer> findAllAtomsAtExactDistance(StereoMolecule mi, int atom, int d, boolean visited[]) {
        if(d==1) {
            Set<Integer> xi = new HashSet<>();
            for(int zi=0;zi<mi.getConnAtoms(atom);zi++) {
                if(!visited[mi.getConnAtom(atom,zi)]) {
                    xi.add(mi.getConnAtom(atom, zi));
                }
            }
            return xi;
        }

        Set<Integer> atoms = new HashSet<>();
        for(int zi=0;zi<mi.getConnAtoms(atom);zi++) {
            int nextAtom = mi.getConnAtom(atom,zi);
            if(!visited[nextAtom]) {
                boolean[] visited_2 = Arrays.copyOf(visited,visited.length);
                visited_2[atom] = true;
                atoms.addAll( findAllAtomsAtExactDistance(mi,nextAtom,d-1,visited_2) );
            }
        }

        return atoms;
    }

    /**
     * Outputs just one conformation per structure
     *
     * @param data
     * @param identifier
     */
    public static void exportSmilesSamples(List<SmilesMolSample3D> data, String identifier, boolean has3d) {
        // now export data into two .npy files:
        String filename_Smiles        = identifier+"_Smiles.npy";
        String filename_SmilesEncoded = identifier+"_SmilesEncoded.npy";
        String filename_DM            = identifier+"_DM.npy";
        String filename_fullDM            = identifier+"_fullDM.npy";
        String filename_ChemInfo             = identifier+"_ChemInfo.npy";
        String filename_distFirst            = identifier+"_distFirst.npy";
        String filename_adjMatrices          = identifier+"_adjMatrices.npy";
        String filename_atomTypeMatrices     = identifier+"_atomTypeMatrices.npy";
        String filename_conformation  = identifier+"_conformation.npy";

        Set<Character> alphabet = new HashSet<>();
        data.stream().forEach(xi -> {for( char ci : xi.smilesWithProblem.toCharArray()){ alphabet.add(ci);}});
        List<Character> list_alphabet = alphabet.stream().sorted().collect(Collectors.toList());
        System.out.println("alphabet size: "+list_alphabet.size());
        System.out.println();
        for(Character ci : list_alphabet) {
            System.out.print(ci);
        }
        System.out.println();
        Map<Character,Integer> charEncoding = new HashMap<>();
        int cnt = 0;
        for(Character ci : list_alphabet) {
            charEncoding.put(ci,cnt);
            cnt++;
        }


        String[]  smilesStrings = new String[data.size()];
        int[][]   smilesEncoded = new int[data.size()][LENGTH_STRING_SEQUENCE];

        int[][][] dmMatrices = new int[data.size()][NUM_DISTANCES][NUM_DISTANCES];
        for(int xi = 0;xi < data.size();xi++) {
            smilesStrings[xi] = data.get(xi).smilesWithProblem;
            for(int xa = 0;xa < NUM_DISTANCES;xa++) {
                for(int xb = 0;xb < NUM_DISTANCES;xb++) {
                    dmMatrices[xi][xa][xb] = data.get(xi).smallDistMatrix[xa][xb];
                }
            }
            for(int zi = 0; zi< LENGTH_STRING_SEQUENCE; zi++) {
                smilesEncoded[xi][zi] = charEncoding.get( data.get(xi).smilesWithProblem.charAt(zi) );
            }
        }

        int[] dmMatrices_unrolled = new int[data.size()*NUM_DISTANCES*NUM_DISTANCES];
        int idx = 0;
        for(int[][] arr2d : dmMatrices) {
            for(int[] arr1d : arr2d) {
                for(int vi : arr1d) {
                    dmMatrices_unrolled[idx] = vi;
                    idx++;
                }
            }
        }

        int[] dmFullMatrices_unrolled = new int[data.size()*NUM_ATOMS*NUM_ATOMS];
        idx = 0;
        for(int xi = 0;xi < data.size();xi++) {
            for (int[] arr1d : data.get(xi).fullDistMatrix) {
                for (int vi : arr1d) {
                    dmFullMatrices_unrolled[idx] = vi;
                    idx++;
                }
            }
        }


        int[] stringsEncoded_unrolled = new int[data.size()* LENGTH_STRING_SEQUENCE];
        idx = 0;

        for(int[] arr1d : smilesEncoded) {
            for(int vi : arr1d) {
                stringsEncoded_unrolled[idx] = vi;
                idx++;
            }
        }

        int[] chemInfo_unrolled = new int[data.size()*2*NUM_ATOMS];
        idx = 0;
        for(int zx = 0 ; zx < data.size(); zx++) {
            for(int vi : data.get(zx).chemInfo) {
                chemInfo_unrolled[idx] = vi;
                idx++;
            }
        }

        int[] distFirstAtom_unrolled = new int[data.size()*NUM_ATOMS];
        idx = 0;
        for(int zx = 0 ; zx < data.size(); zx++) {
            for(int vi : data.get(zx).distForFirst) {
                distFirstAtom_unrolled[idx] = vi;
                idx++;
            }
        }


        float[] conformation_unrolled = new float[data.size()*NUM_ATOMS*3];
        idx = 0;
        if(has3d) {
            for (int zx = 0; zx < data.size(); zx++) {
                for (double[] coords : data.get(zx).conformations.get(0)) {
                    for (double cxi : coords) {
                        conformation_unrolled[idx] = (float) cxi;
                        idx++;
                    }
                }
            }
        }


        boolean[] adjMatricesAtDistUnrolled = new boolean[data.size()*NUM_ATOMS*NUM_ATOMS*8];
        idx = 0;
        for(int zx=0;zx<data.size();zx++) {
            for( boolean[][] arr2d : data.get(zx).distAdjMatrices ) {
                for( boolean[] arr : arr2d) {
                    for(boolean vi : arr) {
                        adjMatricesAtDistUnrolled[idx] = vi;
                        idx++;
                    }
                }
            }
        }

        byte[] atomTypeAtDistMatricesUnrolled = new byte[data.size()*NUM_ATOMS*4*13];
        idx = 0;
        for(int zx=0;zx<data.size();zx++) {
            for( byte[][] arr2d : data.get(zx).atomTypesAtDistanceN ) {
                for( byte[] arr : arr2d) {
                    for(byte vi : arr) {
                        atomTypeAtDistMatricesUnrolled[idx] = vi;
                        idx++;
                    }
                }
            }
        }


        NpyFile.write(Path.of(filename_Smiles), smilesStrings);
        NpyFile.write(Path.of(filename_ChemInfo), chemInfo_unrolled, new int[]{data.size(),NUM_ATOMS*2});
        NpyFile.write(Path.of(filename_DM), dmMatrices_unrolled, new int[]{data.size(),NUM_DISTANCES,NUM_DISTANCES});
        NpyFile.write(Path.of(filename_fullDM), dmFullMatrices_unrolled, new int[]{data.size(),NUM_ATOMS,NUM_ATOMS});
        NpyFile.write(Path.of(filename_SmilesEncoded), stringsEncoded_unrolled, new int[]{data.size(), LENGTH_STRING_SEQUENCE});
        NpyFile.write(Path.of(filename_distFirst), distFirstAtom_unrolled, new int[]{data.size(), NUM_ATOMS});
        NpyFile.write(Path.of(filename_adjMatrices), adjMatricesAtDistUnrolled, new int[]{data.size(), ADJ_MATRICES_AT_DIST_MAX_DISTANCE, NUM_ATOMS, NUM_ATOMS});
        NpyFile.write(Path.of(filename_atomTypeMatrices), atomTypeAtDistMatricesUnrolled, new int[]{data.size(), NUM_ATOMS , ATOM_TYPES_AT_DIST_MAX_DISTANCE, 13});

        if(has3d) {
            NpyFile.write(Path.of(filename_conformation), conformation_unrolled, new int[]{data.size(), 3, NUM_ATOMS});
        }


    }
}