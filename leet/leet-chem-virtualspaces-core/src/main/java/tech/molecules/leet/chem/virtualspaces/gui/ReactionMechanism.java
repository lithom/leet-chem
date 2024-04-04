package tech.molecules.leet.chem.virtualspaces.gui;

import com.actelion.research.chem.IDCodeParser;
import com.actelion.research.chem.io.RXNFileParser;
import com.actelion.research.chem.reaction.Reaction;
import com.actelion.research.chem.reaction.ReactionEncoder;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;

public class ReactionMechanism {
    String filepath;
    Reaction rxn; // just show the toString() value of this

    public ReactionMechanism(String filepath) throws Exception {
        this.filepath = filepath;
        File fi = new File(filepath);
        IDCodeParser icp = new IDCodeParser();

        this.rxn = new Reaction();
        String reactionName = fi.getName().split("\\.")[0];
        this.rxn.setName(reactionName);
        BufferedReader reader = new BufferedReader(new FileReader(filepath));
        //try {
        RXNFileParser rxnParser = new RXNFileParser();
        rxnParser.parse(this.rxn, reader);
        //} catch (Exception e) {
    }

    public String getFilepath() {
        return filepath;
    }

    public void setFilepath(String filepath) {
        this.filepath = filepath;
    }

    public Reaction getRxn() {
        return rxn;
    }

    public void setRxn(Reaction rxn) {
        this.rxn = rxn;
    }

}