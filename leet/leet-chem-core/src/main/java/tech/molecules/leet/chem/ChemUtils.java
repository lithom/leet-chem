package tech.molecules.leet.chem;

import com.actelion.research.chem.*;
import com.actelion.research.chem.coords.CoordinateInventor;
import com.actelion.research.gui.JStructureView;
import tech.molecules.leet.chem.shredder.SynthonShredder;

import javax.swing.*;
import javax.swing.border.LineBorder;
import java.awt.*;
import java.util.*;
import java.util.List;

public class ChemUtils {



    public static String idcodeToSmiles(String idcode) {
        StereoMolecule m = new StereoMolecule();
        IDCodeParser p = new IDCodeParser();
        p.parse(m,idcode);

        String smiles = "exception_in_smiles_creator";
        try{
            IsomericSmilesCreator smiles_creator = new IsomericSmilesCreator(m);
            smiles=smiles_creator.getSmiles();
        }
        catch(Exception ex) {
            System.out.println("Exception in idcodeToSmiles..");
        }
        return smiles;
    }

    public static String stereoMoleculeToSmiles(StereoMolecule sm) {
        IsomericSmilesCreator smiles_creator = new IsomericSmilesCreator(sm);
        return smiles_creator.getSmiles();
    }


    public static StereoMolecule parseIDCode(String idc) {
        IDCodeParser icp = new IDCodeParser();
        StereoMolecule mi = new StereoMolecule();
        icp.parse(mi,idc);
        mi.ensureHelperArrays(Molecule.cHelperCIP);
        return mi;
    }

    public static List<StereoMolecule> parseIDCodes(List<String> idcodes) {
        List<StereoMolecule> parsed = new ArrayList<>();
        IDCodeParser icp = new IDCodeParser();
        for(String si : idcodes) {
            StereoMolecule mi = new StereoMolecule();
            icp.parse(mi, si);
            mi.ensureHelperArrays(Molecule.cHelperCIP);
            parsed.add(mi);
        }
        return parsed;
    }

    public static BitSet findAtomsWithAtomicNo(StereoMolecule mi, int ano) {
        BitSet bs = new BitSet();
        mi.ensureHelperArrays(Molecule.cHelperNeighbours);
        for(int zi=0;zi<mi.getAtoms();zi++) {
            if(mi.getAtomicNo(zi)==ano) {bs.set(zi);}
        }
        return bs;
    }

    /**
     *
     * @param mi
     * @return
     */
    public static List<Integer> findNeighbors(StereoMolecule mi, int a) {
        List<Integer> neighbors = new ArrayList<>();
        //mi.ensureHelperArrays(Molecule.cHelperNeighbours);
        if(mi.getAtoms()<=a) {
            return new ArrayList<>();
        }
        for(int za=0;za<mi.getConnAtoms(a);za++) {
            neighbors.add(mi.getConnAtom(a,za));
        }
        return neighbors;
    }

    public static BitSet findBondsInBetweenAtoms(StereoMolecule mol, BitSet atoms) {
        mol.ensureHelperArrays(Molecule.cHelperCIP);
        BitSet bonds = new BitSet(mol.getBonds());
        // 1. determine bonds:
        for(int zi=0;zi<mol.getBonds();zi++) {
            int baa = mol.getBondAtom(0,zi);
            int bab = mol.getBondAtom(1,zi);
            if( atoms.get(baa) && atoms.get(bab) ) {
                bonds.set(zi,true);
            }
        }
        return bonds;
    }

    public static void highlightBondsInBetweenAtoms(StereoMolecule mol, BitSet atoms) {
        for(int zb=0;zb<mol.getBonds();zb++) {mol.setBondBackgroundHiliting(zb,false);}
        BitSet bs = findBondsInBetweenAtoms(mol,atoms);
        for(int zb=0;zb<mol.getBonds();zb++) {
            if(bs.get(zb)) {
                mol.setBondBackgroundHiliting(zb,true);
            }
        }
    }

    /**
     * Check if any of the bonds share an atom
     *
     * @param mol
     * @param bonds
     * @return
     */
    public static boolean checkForAdjacentBonds(StereoMolecule mol, int[] bonds) {
        boolean found_adj = false;
        BitSet found_atoms = new BitSet(mol.getAtoms());
        for(int zi=0;zi<bonds.length;zi++) {
            int ba = mol.getBondAtom(0,bonds[zi]);
            int bb = mol.getBondAtom(1,bonds[zi]);
            if( found_atoms.get(ba) || found_atoms.get(bb) ) {
                found_adj = true;
                break;
            }
            found_atoms.set(ba,true);
            found_atoms.set(bb,true);
        }
        return found_adj;
    }


    /**
     * Creates a molecule containing the all atoms and bonds
     * in the vicinity of one or multiple atoms.
     *
     * @param mi_pre
     * @param seed_atoms
     * @param region_size
     * @param omit_connectors if true, all atoms >= 88 will not be considered except if they are seed_atoms
     *
     * @return
     */
    public static StereoMolecule createProximalFragment(StereoMolecule mi_pre, List<Integer> seed_atoms, int region_size, boolean omit_connectors, boolean[] neglectAtom) {
        StereoMolecule mi_conn = new StereoMolecule(mi_pre);

        //mi_pre.setFragment(true);
        //mi_conn.removeQueryFeatures(); // !! Remove query features
        // REMOVE ONLY NARROWING QUERY FEATURES:
        //QueryFeatureUtils.removeNarrowingQueryFeatures(mi_conn);

        mi_conn.ensureHelperArrays(Molecule.cHelperCIP);

        // 2. cut out the connector region:
        boolean keep_atoms[] = new boolean[mi_conn.getAtoms()];
        for(int zi=0;zi<mi_conn.getAtoms();zi++) {
            for(int ci : seed_atoms) {
                if(zi==ci) { keep_atoms[zi] = true; break;}
                if(omit_connectors) {
                    if(mi_conn.getAtomicNo(zi)>=88) {
                        continue;
                    }
                }
                // NOTE! this returns -1 if no path is found within connector_region_size (I think..)
                int path_length = mi_conn.getPathLength(ci,zi,region_size,neglectAtom);
                if ( path_length>=0 ){
                    if(path_length<=region_size ) {
                        keep_atoms[zi] = true;
                    }
                }
            }
        }
        StereoMolecule mi_cut = new StereoMolecule();
        mi_cut.setFragment(true);
        mi_conn.copyMoleculeByAtoms(mi_cut,keep_atoms,true,null);
        mi_cut.ensureHelperArrays(Molecule.cHelperCIP);
        if(false) {
            System.out.println("CPF: " + mi_cut.getIDCode());
        }
        return mi_cut;
    }


    public static int hac(StereoMolecule m) {
        m.ensureHelperArrays(Molecule.cHelperNeighbours);
        return m.getAtoms();
    }

    public static int countRingAtoms(StereoMolecule m) {
        m.ensureHelperArrays(Molecule.cHelperNeighbours);
        int ra = 0;
        for(int zi=0;zi<m.getAtoms();zi++) {if(m.isRingAtom(zi)){ra++;}}
        return ra;
    }

    public static int countAromaticAtoms(StereoMolecule m) {
        m.ensureHelperArrays(Molecule.cHelperNeighbours);
        int ra = 0;
        for(int zi=0;zi<m.getAtoms();zi++) {if(m.isAromaticAtom(zi)){ra++;}}
        return ra;
    }

    public static int countAtoms(StereoMolecule m, List<Integer> atomic_numbers) {
        m.ensureHelperArrays(Molecule.cHelperNeighbours);
        int ra = 0;
        for(int zi=0;zi<m.getAtoms();zi++) {if( atomic_numbers.contains(m.getAtomicNo(zi))){ra++;}}
        return ra;
    }

    public static int countRings(StereoMolecule m) {
        m.ensureHelperArrays(Molecule.cHelperCIP);
        RingCollection rings = m.getRingSet();
        return rings.getSize();
    }

    public static int countRingsAromatic(StereoMolecule m) {
        m.ensureHelperArrays(Molecule.cHelperCIP);
        RingCollection rings = m.getRingSet();
        int ra = 0;
        for(int zi=0;zi<rings.getSize();zi++) {if( rings.isAromatic(zi) ){ra++;}}
        return ra;
    }

    public static int countRingsHeteroaromatic(StereoMolecule m) {
        m.ensureHelperArrays(Molecule.cHelperCIP);
        RingCollection rings = m.getRingSet();
        int ra = 0;
        for(int zi=0;zi<rings.getSize();zi++) {
            if( rings.isAromatic(zi) ){
                if( rings.getHeteroPosition(zi)>=0) {
                    ra++;
                }
            }
        }
        return ra;
    }



    public static boolean checkIfAllAtomsAreInSameFragment(StereoMolecule mi, List<Integer> atoms) {
        mi.ensureHelperArrays(Molecule.cHelperCIP);
        if(atoms.size()==0) {return true;} // hmm.. or false? I dont know.. :)
        if(atoms.size()==1) {return true;} // this one is clear I think :)
        int[] frag = mi.getFragmentAtoms(atoms.get(0));
        BitSet bsi = toBitSet(frag);

        BitSet bs_atoms = toBitSet( atoms );

        BitSet bsi_and_bsa = (BitSet) bs_atoms.clone();
        bsi_and_bsa.and(bs_atoms);

        return bs_atoms.cardinality()==bsi_and_bsa.cardinality();
    }


    public static List<Integer> getSelectedAtoms(StereoMolecule m) {
        ArrayList<Integer> sel = new ArrayList<>();
        m.ensureHelperArrays(Molecule.cHelperCIP);
        for(int zi=0;zi<m.getAtoms();zi++) {
            if(m.isSelectedAtom(zi)) { sel.add(zi); }
        }
        return sel;
    }
    public static List<Integer> getSelectedBonds(StereoMolecule m) {
        ArrayList<Integer> sel = new ArrayList<>();
        m.ensureHelperArrays(Molecule.cHelperCIP);
        for(int zi=0;zi<m.getBonds();zi++) {
            if(m.isSelectedBond(zi)) { sel.add(zi); }
        }
        return sel;
    }



    public static BitSet toBitSet(int arr[]) {
        BitSet bs = new BitSet();
        for(int zi=0;zi<arr.length;zi++) { if(arr[zi]>=0){bs.set(arr[zi],true);} }
        return bs;
    }

    public static BitSet toBitSet(List<Integer> val) {
        BitSet bsi = new BitSet();
        for(int zi=0;zi<val.size();zi++) { bsi.set(val.get(zi),true);}
        return bsi;
    }

    public static int[] toIntArray(BitSet bs) {
        return bs.stream().toArray();
    }

    public static int[] toIntArray(List<Integer> list) {
        int[] arr = new int[list.size()];
        for(int zi=0;zi<arr.length;zi++) { arr[zi] = list.get(zi); }
        return arr;
    }

    public static List<Integer> toIntList(BitSet bsi) {
        List<Integer> li = new ArrayList<>();
        bsi.stream().forEach( ii -> li.add(ii) );
        return li;
    }


    public static Map<Integer,Integer> inverseMap(int[] map) {
        Map<Integer,Integer> inv = new HashMap<>();
        for(int zi=0;zi<map.length;zi++) { inv.put( map[zi] , zi ); }
        return inv;
    }

    /**
     * Tries idcode, then smiles. Todo: add more
     * @param str_data
     * @return
     */
    public static StereoMolecule tryParseChemistry(String str_data) {
        StereoMolecule m = new StereoMolecule();
        boolean parsing_success = false;
        try{
            IDCodeParser icp = new IDCodeParser();
            icp.parse(m,str_data);
            parsing_success = true;
        }
        catch(Exception ex) {}

        if(!parsing_success) {
            try{
                SmilesParser smi = new SmilesParser();
                smi.parse(m,str_data);
                parsing_success = true;
            }
            catch(Exception ex) {}
        }

        if(!parsing_success) {
            return null;
        }

        m.ensureHelperArrays(Molecule.cHelperCIP);
        return m;
    }

    public static class DebugOutput {

        static JFrame frame = null;
        static JTabbedPane tp = null;

        public static void plotMolecules(String title, List<String> idcodes, int sx, int sy) {
            plotMolecules(title,idcodes.stream().toArray(String[]::new),sx,sy);
        }


        public static void plotMolecules(String title, String idcodes[], int sx, int sy) {
            plotMolecules(title,parseIDCodes(Arrays.asList(idcodes)).toArray(new StereoMolecule[0]),sx,sy);
        }


        public static void plotMolecules(String title, StereoMolecule mols[], int sx, int sy) {
            plotMolecules(title,mols,sx,sy,true);
        }
        public static void plotMolecules(String title, StereoMolecule mols[], int sx, int sy, boolean inventCoordinates) {

            if (frame == null) {
                JFrame f = new JFrame();
                frame = f;
                tp = new JTabbedPane();
                f.getContentPane().setLayout(new BorderLayout());
                f.getContentPane().add(tp);
                f.setVisible(true);

                int w = Math.min(800, sx * 300);
                int h = Math.min(800, sx * 300);
                f.setSize(w, h);

                f.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
            }

            JScrollPane sp = new JScrollPane();
            JPanel pi = new JPanel();
            pi.setLayout(new GridLayout(sx, sy));
            sp.setViewportView(pi);
            CoordinateInventor ci = new CoordinateInventor();

            for (int zi = 0; zi < mols.length; zi++) {
                StereoMolecule si = mols[zi];
                if(inventCoordinates) {
                    ci.invent(si);
                }
                JStructureView sv = new JStructureView(si);
                //sv.setIDCode(si.getIDCode());
                //sv.set
                pi.add(sv);
                sv.setBorder(new LineBorder(Color.black));
            }

            tp.add(title, pi);
        }


        public static void main(String args[]) {
            String sa = "didHPD@zxHR[Y^FZZX@`";
            String sb = "dmuHPHF`neNdefuQfjjj`B";
            String idcodes[] = new String[]{sa, sb};
            plotMolecules("Out_A", idcodes, 2, 2);
            plotMolecules("Out_b", idcodes, 2, 2);
            plotMolecules("Out_c", idcodes, 2, 2);
        }
    }

}
