package tech.molecules.leet.chem.virtualspaces;

import com.actelion.research.chem.Molecule;
import com.actelion.research.chem.StereoMolecule;
import com.actelion.research.chem.chemicalspaces.ChemicalSpaceCreator;
import com.actelion.research.chem.io.RXNFileParser;
import com.actelion.research.chem.io.SDFileParser;
import com.actelion.research.chem.reaction.Reaction;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.*;

public class SpaceCreation_A {

    public static void main(String[] args) throws FileNotFoundException {

        String pathOutputDir = "C:\\Temp\\virtual_spaces";
        //String pathRxnDir    = "C:\\Temp\\virtual_spaces\\Virtual-Fragment-Spaces-main\\reactions";
        String pathRxnDir    = "C:\\Temp\\virtual_spaces\\Virtual-Fragment-Spaces-main\\reactions_a";
        String[] pathBBFile0   = new String[]{"C:\\Temp\\virtual_spaces\\Virtual-Fragment-Spaces-main\\building_blocks\\Enamine_Building_Blocks.sdf","IDNUMBER"};
        String[] pathBBFile1   = new String[]{"C:\\buildingblocks\\chemdiv\\DC01_400000.sdf","IDNUMBER"};//"C:\\Temp\\virtual_spaces\\Virtual-Fragment-Spaces-main\\building_blocks\\Enamine_Building_Blocks.sdf";
        String[] pathBBFile2   = new String[]{"C:\\buildingblocks\\chemdiv\\DC02_400000.sdf","IDNUMBER"};
        String[] pathBBFile3   = new String[]{"C:\\buildingblocks\\chemdiv\\DC03_241250.sdf","Enamine-ID"};
        List<String[]> paths_BBFiles = new ArrayList<>();
        //paths_BBFiles.add(pathBBFile0);
        paths_BBFiles.add(pathBBFile1);
        //paths_BBFiles.add(pathBBFile2);
        //paths_BBFiles.add(pathBBFile3);
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
        Set<String> bbs = new HashSet<>();
        Map<String, Map<String, List<String>>> bbData = new HashMap<String, Map<String, List<String>>>();

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
                //if(cnt>=200000) {break;}
                String enamineID = parser.getFieldData(idField);
                StereoMolecule bb = parser.getMolecule();
                bb.ensureHelperArrays(Molecule.cHelperParities);
                if (bb.getAtoms() > 20) {
                    continue;
                }
                bbs.add(bb.getIDCode());
                bbData.putIfAbsent(bb.getIDCode(), new HashMap<String, List<String>>());
                Map<String, List<String>> propertyMap = bbData.get(bb.getIDCode());
                propertyMap.putIfAbsent("BB-ID", new ArrayList<>());
                propertyMap.get("BB-ID").add(enamineID);
                cnt++;
                if(bbs.size()>2000) {break;}
            }
        }
        System.out.println("Parsing done.. Compounds: "+bbs.size());
        /*
         * create the space
         */
        ChemicalSpaceCreator2 creator = new ChemicalSpaceCreator2(bbs,reactions,new File(pathOutputDir));
        creator.setBBData(bbData);
        creator.create();

    }

}
