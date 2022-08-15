package tech.molecules.leet.chem.shredder;

import com.actelion.research.chem.Molecule;
import com.actelion.research.chem.StereoMolecule;
import com.actelion.research.chem.hyperspace.SimpleSynthon;
import tech.molecules.leet.chem.QueryFeatureUtils;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;

public class SynthonUtils {



    public static int countNonConnectorAtoms(StereoMolecule mi) {
        mi.ensureHelperArrays(Molecule.cHelperNeighbours);
        int count = 0;
        for(int zi=0;zi<mi.getAtoms();zi++) { count+=mi.getAtomicNo(zi)>88?0:1;}
        return count;
    }

    public static BitSet findConnectorAtoms(StereoMolecule mi) {
        BitSet bs = new BitSet();
        mi.ensureHelperArrays(Molecule.cHelperNeighbours);
        for(int zi=0;zi<mi.getAtoms();zi++) {
            if(mi.getAtomicNo(zi)>=88) {bs.set(zi);}
        }
        return bs;
    }

    public static void removeConnectors(StereoMolecule m) {
        m.ensureHelperArrays(Molecule.cHelperNeighbours);
        for(int zi=0;zi<m.getAtoms();zi++) {
            if(m.getAtomicNo(zi)>=88) { m.markAtomForDeletion(zi); }
        }
        m.deleteMarkedAtomsAndBonds();
        m.ensureHelperArrays(Molecule.cHelperCIP);
    }

    /**
     * Creates the connector proximal fragments.
     *
     * NOTE!! In these fragments, ALL CONNECTORS, i.e. all
     * atoms with atomic no inbetween 92 and 99 WILL BE
     * REPLACED BY uranium, i.e. by 92 !!
     *
     * NOTE!! In these fragments, all NARROWING query features will
     *        be removed. (the rationale is, NARROWING query features
     *        can only RESTRICT the set of compounds, so to be
     *        safe we do not want to have this in a pruning
     *        step)
     *
     * @param mi_pre
     * @param connector_region_size
     * @return
     */
    public static StereoMolecule createConnectorProximalFragment(StereoMolecule mi_pre, int connector_region_size, int[] atommap) {
        StereoMolecule mi_conn = new StereoMolecule(mi_pre);

        //mi_conn.removeQueryFeatures(); // !! Remove query features
        // REMOVE ONLY NARROWING QUERY FEATURES:
        QueryFeatureUtils.removeNarrowingQueryFeatures(mi_conn);


        mi_conn.ensureHelperArrays(Molecule.cHelperCIP);
        //mi_conn.getAtoms();
        // create the connector-near fragment:
        // 1. set all connector atoms to uranium, and store the positions:
        List<Integer> connector_positions = new ArrayList<>();
        for(int zi=0;zi<mi_conn.getAtoms();zi++) {
            int an = mi_conn.getAtomicNo(zi);
            if(an>=92&&an<92+SynthonShredder.MAX_CONNECTORS) {
                connector_positions.add(zi);
                mi_conn.setAtomicNo(zi,92);
            }
        }
        // 2. cut out the connector region, TODO: separately for every connector (such that we do not get glued together connis..)
        boolean keep_atoms[] = new boolean[mi_conn.getAtoms()];
        for(int zi=0;zi<mi_conn.getAtoms();zi++) {
            for(int ci : connector_positions) {
                // NOTE! this returns -1 if no path is found within connector_region_size (I think..)
                int path_length = mi_conn.getPathLength(ci,zi,connector_region_size,null);
                if ( path_length>=0 ){
                    if(path_length<=connector_region_size) {
                        keep_atoms[zi] = true;
                    }
                }
            }
        }
        StereoMolecule mi_cut = new StereoMolecule();
        mi_conn.copyMoleculeByAtoms(mi_cut,keep_atoms,true,atommap);
        mi_cut.ensureHelperArrays(Molecule.cHelperCIP);

        if(false) {
            System.out.println("CPF: " + mi_cut.getIDCode());
        }

        return mi_cut;
    }

    /**
     * Cuts the supplied molecule at specific distance from inner and outer connectors.
     * The positions of the connectors are supplied.
     *
     * @param bdc
     * @param cp_pair_positions
     * @param depthInBondsSynthon
     * @param depthInBondsContext
     * @return
     */
    public static StereoMolecule cutBidirectionalContext(StereoMolecule bdc, int cp_pair_positions[][] , int depthInBondsSynthon, int depthInBondsContext, int[] map_old_to_new) {
        StereoMolecule all_prs = new StereoMolecule();
        all_prs.setFragment(true);

        if(map_old_to_new!=null) {
            for(int zi=0;zi<map_old_to_new.length;zi++){ map_old_to_new[zi]=-1;}
        }

        // create fragments
        for(int zp=0;zp<cp_pair_positions.length;zp++) {

            int pos_inner = cp_pair_positions[zp][0];
            int pos_outer =cp_pair_positions[zp][1];
            StereoMolecule mi_conn = bdc;//new StereoMolecule(lc);
            //mi_conn.ensureHelperArrays(Molecule.cHelperCIP);
            mi_conn.ensureHelperArrays(Molecule.cHelperCIP);

            // 2. cut out the connector region:
            boolean keep_atoms[] = new boolean[mi_conn.getAtoms()];
            for(int zi=0;zi<mi_conn.getAtoms();zi++) {
                //for(int ci : seed_atoms) {
                if(zi==pos_inner) { keep_atoms[zi] = true; continue;}
                if(zi==pos_outer) { keep_atoms[zi] = true; continue;}
                // NOTE! this returns -1 if no path is found within connector_region_size (I think..)
                int path_length_inner = mi_conn.getPathLength(pos_inner,zi);
                int path_length_outer = mi_conn.getPathLength(pos_outer,zi);


                if(path_length_inner<0 && path_length_outer<0) {
                    continue;
                }

                if ( path_length_inner>=0  && path_length_inner<path_length_outer){
                    if(path_length_inner<=depthInBondsSynthon) {
                        keep_atoms[zi] = true;
                    }
                }
                if ( path_length_outer>=0  && path_length_outer<path_length_inner){
                    if(path_length_outer<=depthInBondsContext) {
                        keep_atoms[zi] = true;
                    }
                }
                //}
            }
            int c_amap1[] = new int[mi_conn.getAtoms()];
            StereoMolecule mi_cut = new StereoMolecule();
            mi_cut.setFragment(true);
            mi_conn.copyMoleculeByAtoms(mi_cut,keep_atoms,true,c_amap1);
            mi_cut.ensureHelperArrays(Molecule.cHelperCIP);
            if(false) {
                System.out.println("CPF: " + mi_cut.getIDCode());
            }
            int c_amap2[] = new int[mi_cut.getAtoms()];
            all_prs.addFragment(mi_cut,0,c_amap2);

            for(int zi=0;zi<bdc.getAtoms();zi++) {
                if(map_old_to_new!=null && map_old_to_new.length>zi) {
                    if(c_amap1[zi]>=0) {
                        //c_amap2[ c_amap1[zi] ] = zi;
                        map_old_to_new[zi] = c_amap2[ c_amap1[zi] ];
                    }
                }
            }
        }
        all_prs.ensureHelperArrays(Molecule.cHelperCIP);
        return all_prs;
    }

}
