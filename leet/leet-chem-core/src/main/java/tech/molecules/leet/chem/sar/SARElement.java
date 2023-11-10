package tech.molecules.leet.chem.sar;

import com.actelion.research.chem.Molecule;
import com.actelion.research.chem.StereoMolecule;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * All is done via labeling:
 *
 * Atom can be labeled as follows:
 * "R-xxx" defines r-group xxx
 * "R-xxx{c(13)}" defines r-group xxx and requires connection c(13)
 * no label : "prevent further substitution"
 *
 *
 * Now the following logic applies: if there exists at least one part of the molecule that is not well-defined, then
 * this scaffold part is considered as "R-group-like", because there may exist different versions of it.
 * Also in case that there are bridge bonds or multi-atoms, then the scaffold itself
 * is considered as "R-group-like"
 *
 */

public class SARElement {
    private StereoMolecule mol;

    public SARElement(StereoMolecule mol) {
        this.mol = mol;
        this.mol.ensureHelperArrays(Molecule.cHelperCIP);
    }

    public StereoMolecule getMol() {
        return this.mol;
    }

    /**
     * RX:  extendable (r-group inducing atom)
     * SX: extendable (scaffold inducing atom)
     * FIX: prevent further subsitution
     */
    public enum ScaffoldAtomType {RX,SX,FIX}

    public static class ScaffoldAtomConf {
        public final ScaffoldAtomType type;
        public final String           label;
        public final int              connectionConstraint;
        public String toString() {
            String xa = null;
            switch(type) {
                case RX:
                    xa = "R-"+label;
                    if(connectionConstraint>=0) { xa += "{c("+connectionConstraint+")}";}
                    break;
                case SX:
                    xa = "SX";
                case FIX:
                    xa = "";
            }
            return xa;
        }

        public ScaffoldAtomConf(ScaffoldAtomType type, String label, int connectionConstraint) {
            this.type = type;
            this.label = label;
            this.connectionConstraint = connectionConstraint;
        }

        public ScaffoldAtomConf(String xa) {
            xa.trim();
            String xlabel = null;
            int xConnConstraint = -1;
            ScaffoldAtomType sat = null;
            if(xa == null) {
                sat = ScaffoldAtomType.FIX;
            }
            else if(xa.isEmpty()) {
                sat = ScaffoldAtomType.FIX;
            }
            else if(xa.startsWith("R-")) {
                sat = ScaffoldAtomType.RX;
                if(!xa.contains("{")) {
                    xlabel = xa.substring(2);
                }
                else {
                    String input = "R-xxx{c(42)}";
                    // Define the regular expression pattern
                    String regex = "R-([A-Za-z0-9]+)\\{c\\((\\d+)\\)}";
                    // Create a Pattern object
                    Pattern pattern = Pattern.compile(regex);
                    // Create a Matcher object
                    Matcher matcher = pattern.matcher(input);
                    // Find the matching pattern
                    if (matcher.find()) {
                        // Extract the matched alphanumeric string from the first capturing group
                        String matchedString = matcher.group(1);
                        // Extract the matched number from the second capturing group
                        String matchedNumber = matcher.group(2);
                        // Convert the matched number to an integer
                        int dd = Integer.parseInt(matchedNumber);
                        // Print the extracted string and number
                        System.out.println("Extracted string: " + matchedString);
                        System.out.println("Extracted number: " + dd);
                        xlabel = matchedString;
                        xConnConstraint = dd;
                    } else {
                        System.out.println("No match found.");
                    }
                }
            }
            else if(xa.startsWith("SX")) {
                sat = ScaffoldAtomType.SX;
                xlabel = "";
            }
            else {
                System.out.println("[WARN] cannot really interpret atom label: "+xa);
                sat = ScaffoldAtomType.FIX;
                xlabel = "";
            }

            //this(sat,xlabel,xConnConstraint);
            this.type = sat;
            this.label = xlabel;
            this.connectionConstraint = xConnConstraint;
        }
    }

    /**
     * true if parts of this scaffold are variable and not r-groups.
     * I.e. if the scaffold contains SX atoms or atoms with multiple allowed atoms or bridge bonds..
     *
     * @return
     */
    public boolean isRGroupLike() {
        boolean isRGroupLike = false;

        // check atoms:
        for(int zi=0;zi<mol.getAtoms();zi++) {
            ScaffoldAtomConf sac = getAtomConf(mol,zi);
            if(sac.type==ScaffoldAtomType.SX) {
                isRGroupLike = true;
                return true;
            }
            else { // check if multiple atomic numbers allowed:
                int[] atomlist_a = mol.getAtomList(zi);
                if(atomlist_a!=null && atomlist_a.length>0) {
                    isRGroupLike = true;
                    return true;
                }
            }
        }

        // check bonds: (search for bridge bonds)
        for(int zi=0;zi<mol.getBonds();zi++) {
            if( mol.isBondBridge(zi) ) {
                isRGroupLike = true;
                return true;
            }
        }

        return isRGroupLike;
    }

    public ScaffoldAtomConf getAtomConf(int atom) {
        return getAtomConf(this.mol,atom);
    }

    public ScaffoldAtomType getScaffoldAtomType(int atom) {
        return getAtomConf(this.mol,atom).type;
    }

    /**
     * Resolves the configuration of a specific atom
     *
     * @param mi
     * @param atom
     */
    public static ScaffoldAtomConf getAtomConf(StereoMolecule mi, int atom) {
        return new ScaffoldAtomConf(mi.getAtomCustomLabel(atom));
    }

    /**
     *
     * @return all non labeled atoms set to prevent further substitution
     */
    public StereoMolecule getQueryFrag() {
        StereoMolecule qf = new StereoMolecule(mol);
        qf.setFragment(true);
        qf.ensureHelperArrays(Molecule.cHelperCIP);
        for(int zi=0;zi<mol.getAtoms();zi++) {
            if( qf.getAtomCustomLabel(zi) == null || qf.getAtomCustomLabel(zi).equals("") ) {
                qf.setAtomQueryFeature(zi,StereoMolecule.cAtomQFNoMoreNeighbours,true);
            }
            else {
                qf.setAtomQueryFeature(zi,StereoMolecule.cAtomQFNoMoreNeighbours,false);
            }
        }
        return qf;
    }
}
