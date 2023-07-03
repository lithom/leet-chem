package tech.molecules.leet.chem.sar;

import com.actelion.research.chem.Canonizer;
import com.actelion.research.chem.Molecule;
import com.actelion.research.chem.SSSearcher;
import com.actelion.research.chem.StereoMolecule;
import com.actelion.research.chem.coords.CoordinateInventor;
import com.actelion.research.chem.coords.InventorTemplate;
import com.actelion.research.chem.descriptor.DescriptorHandlerLongFFP512;
import org.apache.commons.lang3.tuple.Pair;
import tech.molecules.leet.chem.ChemUtils;

import java.util.*;
import java.util.stream.Collectors;

public class SimpleSARDecomposition {

    public static class FragmentSeed {
        final String label;
        final List<Integer> atoms;
        public FragmentSeed(String label, List<Integer> atoms) {
            this.label = label;
            this.atoms = atoms;
        }
    }

    public static class MatchedFragment {
        public final String label;
        public final FragmentSeed fragmentSeed;

        public final StereoMolecule mol;
        public final StereoMolecule matchedFrag;
        public final StereoMolecule sar;

        public final int[] mapFragToMol;
        public final int[] mapSarToFrag;

        public MatchedFragment(String label, FragmentSeed fragmentSeed, StereoMolecule mol, StereoMolecule matchedFrag, StereoMolecule sar, int[] mapFragToMol, int[] mapSarToFrag) {
            this.label = label;
            this.fragmentSeed = fragmentSeed;
            this.mol = mol;
            this.matchedFrag = matchedFrag;
            this.sar = sar;
            this.mapFragToMol = mapFragToMol;
            this.mapSarToFrag = mapSarToFrag;
        }
    }

    public static class SimpleSARResult extends HashMap<String,MatchedFragment> {
        private String[] structure;
        public SimpleSARResult(String[] structure) {
            this.structure = structure;
        }
        public String[] getStructure() {
            return structure;
        }
    }

    public static List<SimpleSARResult> matchSimpleSAR(StereoMolecule sarFragment, StereoMolecule mol_a) {
        sarFragment.ensureHelperArrays(Molecule.cHelperCIP);
        mol_a.ensureHelperArrays(Molecule.cHelperCIP);

        SSSearcher sss = new SSSearcher();
        sss.setMol(sarFragment,mol_a);
        sss.findFragmentInMolecule();

        List<int[]> matches = sss.getMatchList();
        System.out.println("matches: "+matches.size());

        List<SimpleSARResult> results = new ArrayList<>();

        for(int zi=0;zi<matches.size();zi++) {
            int[] mi = matches.get(zi);
            Set<Integer> sar_atoms = new HashSet<>();
            Arrays.stream(mi).forEach( xi -> sar_atoms.add(xi) );

            // invent coordinates based on template?
            if(true) {
                CoordinateInventor ci = new CoordinateInventor();
                DescriptorHandlerLongFFP512 ffp = new DescriptorHandlerLongFFP512();
                ci.setCustomTemplateList( Collections.singletonList(new InventorTemplate( sarFragment , ffp.createDescriptor(sarFragment) ,true)) );
                ci.invent(mol_a);
            }

            // process match:
            Map<String,List<Integer>> labeledAtoms = new HashMap<>();
            for(int za=0;za<sarFragment.getAtoms();za++) {
                String li = extractLabel(sarFragment,za);
                if(li!=null) {
                    labeledAtoms.putIfAbsent(li,new ArrayList<>());
                    labeledAtoms.get(li).add(za);
                }
            }

            List<FragmentSeed> seeds = new ArrayList<>();
            for(Map.Entry<String,List<Integer>> ei : labeledAtoms.entrySet()) {
                seeds.add(new FragmentSeed(ei.getKey(),ei.getValue()));
            }




            // process seeds. I.e. cut all bonds in between the seed atoms and other atoms that are part
            // of the SAR scaffold, but differently labeled
            // AND: compute coloring of seed atoms that neighbor other parts of the scaffold..
            // Map<FragmentSeed, Map<Integer,Integer>> color_border_atoms = new HashMap<>();
            List<MatchedFragment> matchedFrags = new ArrayList<>();
            for(FragmentSeed si : seeds) {
                int col_cnt = 0;
                int[] atom_colors = new int[]{Molecule.cAtomColorOrange,Molecule.cAtomColorMagenta, Molecule.cAtomColorDarkRed, Molecule.cAtomColorDarkGreen,
                                              Molecule.cAtomColorBlue,Molecule.cAtomColorGreen, Molecule.cAtomColorRed};
                StereoMolecule mol_i = new StereoMolecule(mol_a);
                mol_i.ensureHelperArrays(Molecule.cHelperCIP);
                Set<Integer> labeled_atoms = new HashSet<>();
                si.atoms.stream().forEach( xi -> labeled_atoms.add( mi[xi] ));
                // cut:
                for(int bi=0;bi<mol_i.getBonds();bi++) {
                    int at_a = mol_i.getBondAtom(0,bi);
                    int at_b = mol_i.getBondAtom(1,bi);
                    if(sar_atoms.contains(at_a) && sar_atoms.contains(at_b)) {
                        int num_labeled = ((labeled_atoms.contains(at_a)) ? 1 : 0) + ((labeled_atoms.contains(at_b)) ? 1 : 0);
                        if (num_labeled == 1) {
                            // cut:
                            //mol_i.deleteBond(bi);
                            mol_i.markBondForDeletion(bi);
                            // color the border-connecting atom to uniquely define it.
                            if(labeled_atoms.contains(at_a)) { mol_i.setAtomColor(at_a,atom_colors[col_cnt%atom_colors.length]); mol_i.setAtomCustomLabel(at_a,"P"+col_cnt); }
                            if(labeled_atoms.contains(at_b)) { mol_i.setAtomColor(at_b,atom_colors[col_cnt%atom_colors.length]); mol_i.setAtomCustomLabel(at_b,"P"+col_cnt);}
                            col_cnt++;
                        }
                    }
                }
                mol_i.deleteMarkedAtomsAndBonds();
                mol_i.ensureHelperArrays(Molecule.cHelperCIP);
                // now collect all fragments that contain labeled atoms (should be 1, but other is also fine maybe)
                Set<Set<Integer>> fragments = new HashSet<>();
                for(int xi : labeled_atoms) {
                    int[] fxi = mol_i.getFragmentAtoms(xi);
                    fragments.add( Arrays.stream(fxi).boxed().collect(Collectors.toSet()));
                }

                StereoMolecule matchedFrag = new StereoMolecule();
                if(fragments.size()>1) {
                    System.out.println("[WARN] more than one result fragment?");
                }
                int[] all_mapped_atoms_old_to_destmol = new int[mol_i.getAtoms()];
                Arrays.fill(all_mapped_atoms_old_to_destmol,-1);
                for(Set<Integer> sfi : fragments) {
                    boolean[] include_atoms = new boolean[mol_i.getAtoms()];
                    sfi.stream().forEach( xi -> include_atoms[xi] = true);
                    int[] atom_map_old_to_destmol = new int[mol_i.getAtoms()];
                    mol_i.copyMoleculeByAtoms(matchedFrag, include_atoms, true,atom_map_old_to_destmol);
                    for(int zm = 0; zm<atom_map_old_to_destmol.length; zm++) {
                        if(atom_map_old_to_destmol[zm]>=0) { all_mapped_atoms_old_to_destmol[zm] = atom_map_old_to_destmol[zm];}
                    }
                }
                matchedFrag.ensureHelperArrays(Molecule.cHelperCIP);
                int[] map_frag_to_mol = ChemUtils.inverseMap2(all_mapped_atoms_old_to_destmol,mol_a.getAtoms(),matchedFrag.getAtoms());
                int[] map_sar_to_frag = new int[mi.length];
                for(int zj=0;zj<mi.length;zj++) {
                    map_sar_to_frag[zj] = -1;
                    int xi = all_mapped_atoms_old_to_destmol[ mi[zj] ];
                    if(xi >= 0) {map_sar_to_frag[zj] = xi;}
                }

                matchedFrags.add(new MatchedFragment(si.label,si,mol_a,matchedFrag,sarFragment,map_frag_to_mol,map_sar_to_frag));
            }
            SimpleSARResult ssri = new SimpleSARResult( new String[]{ mol_a.getIDCode() , mol_a.getIDCoordinates() } );
            matchedFrags.stream().forEach( xi -> ssri.put(xi.label,xi) );
            results.add(ssri);
        }
        return results;
    }

    public static String extractLabel(StereoMolecule ma, int atom) {
        String la = ma.getAtomCustomLabel(atom);
        if( la !=null && !la.isEmpty()) {
            return la;
        }
        return null;
    }

    public static List<String> extractAllLabels(StereoMolecule ma) {
        List<String> labels = new ArrayList<>();
        for(int zi=0;zi<ma.getAtoms();zi++) {
            String eli = extractLabel(ma,zi);
            if(eli!=null && !eli.isEmpty()) {
                labels.add(eli);
            }
        }
        return labels;
    }

    public static void main(String args[]) {
        StereoMolecule sar_a = ChemUtils.parseIDCode("fdiP`@DD@iInYfw_myjjA`b`@NT^PuT^XEwzZIBiZ^\\o}Rr~XloSKKFCqdmONW~pQYYXpYYYXpSYYXp[YYXpWTeen~\\GvVVLDMIY[ogAcrVggKtWYO|h^`");
        StereoMolecule mol_a = ChemUtils.parseIDCode("ejUQDH@AKMk`XT\\SG@ICHhdhXhhhiMkHhcDdiUjF@hfenemkd\\X@JjYjjjZijVbjhNB@@");

        List<SimpleSARResult> results = SimpleSARDecomposition.matchSimpleSAR(sar_a,mol_a);
        for(SimpleSARResult ri : results) {
            for(Map.Entry<String,MatchedFragment> xi : ri.entrySet()) {
                System.out.println(xi.getKey()+ " -> "+xi.getValue().matchedFrag.getIDCode());
            }
        }
        System.out.println("mkay");

        System.out.println("try to reassemble..");

        List<MatchedFragment> frags = new ArrayList<>( results.get(0).values() );
        StereoMolecule mol2         = mergeParts(frags);
        System.out.println(mol2.getIDCode());
        System.out.println("mkay2");
    }

    public static StereoMolecule mergeParts(List<MatchedFragment> fragments) {
        if(fragments.size()==0) {return new StereoMolecule();}

        StereoMolecule scaffold = new StereoMolecule(fragments.get(0).sar);
        scaffold.ensureHelperArrays(Molecule.cHelperCIP);

        for(int zi=0;zi< fragments.size();zi++) {
            int[] map_sar_to_fi = fragments.get(zi).mapSarToFrag;
            StereoMolecule mfi = new StereoMolecule( fragments.get(zi).matchedFrag );
            mfi.ensureHelperArrays(Molecule.cHelperCIP);
            int map_fi_to_assembled[] = new int[mfi.getAtoms()];
            scaffold.addFragment(mfi,0,map_fi_to_assembled);
            scaffold.ensureHelperArrays(Molecule.cHelperCIP);

            // now: go over all scaffold atoms and connect its neighbors with all non-fragment atoms:
            Map<Integer,Integer> map_scaffold_atoms_of_frag_to_scaffold_atom = new HashMap<>();
            Set<Integer> scaffold_atoms_of_frag = new HashSet<>();
            for(int zj=0;zj<map_sar_to_fi.length;zj++) {
                if(map_sar_to_fi[zj]>=0) {
                    scaffold_atoms_of_frag.add( map_fi_to_assembled[ map_sar_to_fi[zj] ] );
                    map_scaffold_atoms_of_frag_to_scaffold_atom.put( map_fi_to_assembled[ map_sar_to_fi[zj] ] , zj );
                    scaffold.setAtomColor(zj,Molecule.cAtomColorGreen);
                    scaffold.setAtomColor(map_fi_to_assembled[ map_sar_to_fi[zj] ],Molecule.cAtomColorMagenta);
                }
            }

            System.out.println("now: "+(new Canonizer(scaffold)).getIDCode());
            ChemUtils.DebugOutput.plotMolecules("a",new StereoMolecule[]{scaffold},1,1);

            for(int zj=0;zj<map_fi_to_assembled.length;zj++) {
                int ai = map_fi_to_assembled[zj];
                if( scaffold_atoms_of_frag.contains(ai) ) {
                    // now we have to "merge"
                    // i.e.
                    // 1. find all neighbors that are not scaffold_atoms_of_frag and store their bond information (i.e. bond to ai)
                    // 2. remove atom
                    // 3. connect neighbors to scaffold atom (zj)

                    // <atom,bondtype>
                    List<Pair<Integer,Integer>> connections = new ArrayList<>();
                    for(int za=0;za<scaffold.getConnAtoms(ai);za++) {
                        int na = scaffold.getConnAtom(ai,za);
                        if(!scaffold_atoms_of_frag.contains(na)) {
                            int bondtype = scaffold.getBondType(scaffold.getBond(ai,na));
                            connections.add(Pair.of(na,bondtype));
                        }
                    }
                    // we dont delete atoms, we do this just at the end.. marking leads to premature deletion when
                    // bonds are deleted I guess..
                    //scaffold.markAtomForDeletion(ai);

                    // add bonds:
                    for(Pair<Integer,Integer> pxi : connections) {
                        // delete old bond:
                        scaffold.deleteBond(scaffold.getBond(ai,pxi.getLeft()));
                        scaffold.addBond( map_scaffold_atoms_of_frag_to_scaffold_atom.get(ai),pxi.getLeft(),pxi.getRight());
                        scaffold.ensureHelperArrays(Molecule.cHelperNeighbours);
                    }

                    //scaffold.deleteMarkedAtomsAndBonds();
                    scaffold.ensureHelperArrays(Molecule.cHelperCIP);

                    System.out.println("now: "+new StereoMolecule(scaffold).getIDCode());
                    System.out.println("bla");
                }
            }
        }
        scaffold.ensureHelperArrays(Molecule.cHelperCIP);
        return scaffold;
    }

}
