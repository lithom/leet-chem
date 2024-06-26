package tech.molecules.leet.chem.virtualspaces;

import com.actelion.research.chem.Molecule;
import com.actelion.research.chem.StereoMolecule;
import com.actelion.research.chem.chemicalspaces.ChemicalSpaceCreator;
import com.actelion.research.chem.io.RXNFileParser;
import com.actelion.research.chem.io.SDFileParser;
import com.actelion.research.chem.reaction.Reaction;
import org.apache.commons.lang3.tuple.Pair;
import tech.molecules.leet.chem.virtualspaces.gui.LoadedBB;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.*;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Collectors;

public class SpaceCreation_A {

    /**
     * Actually we select the N smallest bbs that we find in all of the considered input files.
     *
     * @param args
     * @throws FileNotFoundException
     */
    public static void main(String[] args) throws FileNotFoundException {

        //String pathOutputDir = "C:\\Temp\\virtual_spaces_divchem_64k";
        String pathOutputDir = "C:\\temp\\virtual_spaces\\divchem_v2size18  _xxl_s2_xxl";
        //String pathRxnDir    = "C:\\Temp\\virtual_spaces\\Virtual-Fragment-Spaces-main\\reactions";
        String pathRxnDir    = "C:\\datasets\\reactions\\reactions_b";//"C:\\Temp\\virtual_spaces\\Virtual-Fragment-Spaces-main\\reactions_a";
        //String pathRxnDir    = "C:\\datasets\\reactions\\reactions";//"C:\\Temp\\virtual_spaces\\Virtual-Fragment-Spaces-main\\reactions_a";

        String[] pathBBFile0   = new String[]{"C:\\Temp\\virtual_spaces\\Virtual-Fragment-Spaces-main\\building_blocks\\Enamine_Building_Blocks.sdf","IDNUMBER"};
        String path_BB_chemdiv = "C:\\datasets\\buildingblocks\\divchem";
        String[] pathBBFile1   = new String[]{path_BB_chemdiv+"\\DC01_400000.sdf","IDNUMBER"};//"C:\\Temp\\virtual_spaces\\Virtual-Fragment-Spaces-main\\building_blocks\\Enamine_Building_Blocks.sdf";
        String[] pathBBFile2   = new String[]{path_BB_chemdiv+"\\DC02_400000.sdf","IDNUMBER"};
        String[] pathBBFile3   = new String[]{path_BB_chemdiv+"\\DC03_241250.sdf","IDNUMBER"};
        List<String[]> paths_BBFiles = new ArrayList<>();
        //paths_BBFiles.add(pathBBFile0);
        paths_BBFiles.add(pathBBFile1);
        paths_BBFiles.add(pathBBFile2);
        paths_BBFiles.add(pathBBFile3);
        //String idfieldName   = "IDNUMBER";//"Enamine-ID";

        RXNFileParser rxnParser = new RXNFileParser();
        File rxnDir = new File(pathRxnDir);
        List<Reaction> reactions = new ArrayList<>();
        /*
         * parsing the files with the reaction definitions (.rxn files)
         */
        for(File reactionFile : rxnDir.listFiles()) {
            if(!reactionFile.getName().endsWith(".rxn"))
                continue;
            Reaction reaction = new Reaction();
            String reactionName = reactionFile.getName().split("\\.")[0];
            reaction.setName(reactionName);
            BufferedReader reader = new BufferedReader(new FileReader(reactionFile));
            try {
                rxnParser.parse(reaction, reader);
            }
            catch(Exception e) {
                continue;
            }
            reactions.add(reaction);
        }

        /*
         * parsing the files with the building blocks
         */
        Set<Pair<String,Integer>> bbs_1 = new HashSet<>();
        Map<String, Map<String, List<String>>> bbData = new HashMap<String, Map<String, List<String>>>();

        //List<Pair<String,Integer>> sortedMols = new ArrayList<>();

        for(String[] pathBBFileData : paths_BBFiles) {
            String pathBBFile = pathBBFileData[0];
            String idfieldName = pathBBFileData[1];
            SDFileParser parser = new SDFileParser(pathBBFile);
            String[] columns = parser.getFieldNames();
            parser.close();
            parser = new SDFileParser(pathBBFile, columns);
            int idField = parser.getFieldIndex(idfieldName);
            int cnt = 0;
            while (parser.next()) {
                //if(cnt>=20000) {break;}
                String enamineID = parser.getFieldData(idField);
                StereoMolecule bb = parser.getMolecule();
                bb.ensureHelperArrays(Molecule.cHelperParities);
                //if (bb.getAtoms() > 20) {
                if (bb.getAtoms() > 18) {
                    continue;
                }
                bbs_1.add(Pair.of(bb.getIDCode(),bb.getAtoms()));
                bbData.putIfAbsent(bb.getIDCode(), new HashMap<String, List<String>>());
                Map<String, List<String>> propertyMap = bbData.get(bb.getIDCode());
                propertyMap.putIfAbsent("BB-ID", new ArrayList<>());
                propertyMap.get("BB-ID").add(enamineID);
                cnt++;
                //if(bbs.size()>2000) {break;}
            }
        }

        System.out.println("Parsing done.. Compounds_total: "+bbs_1.size());
        // Now take the N smallest:
        List<Pair<String,Integer>> bbs_1_selected = bbs_1.stream().sorted( (x,y) -> Integer.compare(x.getRight(),y.getRight()) ).collect(Collectors.toList());//.subList(0,64000);//bbs_1.stream().sorted( (x,y) -> Integer.compare(x.getRight(),y.getRight()) ).collect(Collectors.toList()).subList(0,64000);
        Set<String> bbs = new HashSet<>( bbs_1_selected.stream().map(xi -> xi.getLeft()).collect(Collectors.toList()) );

        /*
         * create the space
         */
        ChemicalSpaceCreator2 creator = new ChemicalSpaceCreator2(bbs,reactions,new File(pathOutputDir));
        creator.setBBData(bbData);
        creator.create();

    }

    /**
     * Initializes the space creator with the building blocks.
     * The molid property is set to "BB-ID".
     *
     * @param bbs
     * @param reactions
     * @param outputDir
     * @param bbFilters return true if bb should be used, false if bb should be filtered out
     * @return
     */
    public static ChemicalSpaceCreator2 createSpaceCreator(List<LoadedBB> bbs, List<Reaction> reactions, File outputDir, List<Function<LoadedBB,Boolean>> bbFilters, Consumer<String> statusOutput) {

        if(bbFilters == null) {bbFilters = new ArrayList<>();}

        Map<String, Map<String, List<String>>> bbData = new HashMap<>();
        List<String> consideredBBs = new ArrayList<>();
        for(LoadedBB bb : bbs) {
            bbData.putIfAbsent(bb.getIdcode(), new HashMap<String, List<String>>());
            Map<String, List<String>> propertyMap = bbData.get(bb.getIdcode());
            propertyMap.putIfAbsent("BB-ID", new ArrayList<>());
            propertyMap.get("BB-ID").add(bb.getMolid());

            // check if we put it:
            boolean addThisOne = true;
            for(Function<LoadedBB,Boolean> filter : bbFilters) {
                boolean ok = filter.apply(bb);
                if(!ok) {addThisOne = false;}
                break;
            }
            if(addThisOne) { consideredBBs.add(bb.getIdcode()); }
        }
        statusOutput.accept("BBs: "+consideredBBs.size() + " Rxns: "+reactions.size());
        ChemicalSpaceCreator2 creator = new ChemicalSpaceCreator2(new HashSet<>(consideredBBs),reactions,outputDir,statusOutput);
        creator.setBBData(bbData);
        return creator;
    }

}
