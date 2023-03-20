package tech.molecules.deep;

import com.actelion.research.chem.Molecule;
import com.actelion.research.chem.StereoMolecule;
import tech.molecules.leet.chem.ChemUtils;

import java.util.*;


/**
 * Format: Each atom is encoded with the same information:
 *
 * 1. One-Hot Atom
 *
 * Information about bonds to all n other atoms. For pair of atom i and j:
 * if i and j are the same, or if i and j are not connected: all zeros
 * otherwise:
 *
 * 1.
 *
 *
 *
 *
 */
public class MolEncoder {

    private StereoMolecule m;
    int N = 32;
    int[] mappedAtomicNo = new int[] {1,6,7,8,9,15,16,17,35};

    Map<Integer,BitSet> encoding_AtomOneHot = new HashMap<>();

    Map<Integer,BitSet> encoding_BondTypeSimple = new HashMap<>();

    public MolEncoder() {
        init();
    }

    private void init() {
        for(int zi=0;zi<mappedAtomicNo.length;zi++) {
            BitSet bsi = new BitSet();
            bsi.set(zi);
            encoding_AtomOneHot.put(mappedAtomicNo[zi],bsi);
        }

        /**
         * BondTypeSimple
         */
        //encoding_BondTypeSimple = new HashMap<>();
        //encoding_BondTypeSimple.put(Molecule.cBondTypeSingle,oneHot(0,7));
        //encoding_BondTypeSimple.put(Molecule.cBondTypeDouble,oneHot(1,7));
        //encoding_BondTypeSimple.put(Molecule.cBondTypeTriple,oneHot(2,7));
        //encoding_BondTypeSimple.put(Molecule.cBondTypeDelocalized,oneHot(3,7));
        //encoding_BondTypeSimple.put(Molecule.cBondTypeUp,oneHot(4,7));
        //encoding_BondTypeSimple.put(Molecule.cBondTypeDown,oneHot(5,7));
        //encoding_BondTypeSimple.put(Molecule.cBondType,oneHot(6,7));
        //encoding_bondTypeSimple.put(Molecule.cBond;

    }
    public void setMolecule(StereoMolecule m) {
        this.m = m;
    }


    public List<BitSet> encodeMolecule() {
        m.ensureHelperArrays(Molecule.cHelperCIP);
        int[] permutation = new int[N];
        for(int zi=0;zi<permutation.length;zi++){permutation[zi]=zi;}
        return encodeMolecule(permutation);
    }

    public List<BitSet> encodeMolecule(int[] permutation) {
        List<BitSet> bs = new ArrayList<>();
        for(int zi=0;zi<permutation.length;zi++) {
            BitSet bsi = encodeAtomWithConnections(zi,permutation);
            bs.add(bsi);
        }
        return bs;
    }

    public int getMoleculeEncodingLength() {
        return N * ( getAtomEncodingLength() + N*getConnectionEncodingLengthPerConnection() );
    }

    public int getAtomEncodingLength() {
        int encoding_length = 0;
        encoding_length += mappedAtomicNo.length;
        encoding_length += 2; // one hot stereo
        return encoding_length;
    }

    public int getConnectionEncodingLengthPerConnection() {
        int encoding_length = 1;

        return 7;
    }

    public BitSet encodeAtomWithConnections(int zi, int[] permutation) {
        if(permutation[zi]>= m.getAtoms()) {
            return new BitSet(getAtomEncodingLength() + N * getConnectionEncodingLengthPerConnection());
        }

        BitSet b_all = new BitSet( getAtomEncodingLength() + N * getConnectionEncodingLengthPerConnection() );
        BitSet b_i = encodeAtom( permutation[zi] );
        b_all.or( b_i );

        for(int zj=0;zj<N;zj++) {
            BitSet b_ij = encodeConnection(permutation[zi],permutation[zj]);
            int ni = getAtomEncodingLength() + zj*getConnectionEncodingLengthPerConnection();
            for(int zx=0;zx<getConnectionEncodingLengthPerConnection();zx++) {
                b_all.set( ni+zx , b_ij.get(zx) );
            }
        }
        return b_all;
    }

    public BitSet encodeAtom(int zi) {
        BitSet ba = encoding_AtomOneHot.get(m.getAtomicNo(zi));
        //BitSet stereo = new BitSet();
        if( m.getAtomParity(zi)==Molecule.cAtomParity1) {ba.set(mappedAtomicNo.length+0);}
        if( m.getAtomParity(zi)==Molecule.cAtomParity2) {ba.set(mappedAtomicNo.length+1);}
        return ba;
    }

    public BitSet encodeConnection(int zi, int zj) {
        if(zi==zj || m.getBond(zi,zj) < 0) {
            return new BitSet(getConnectionEncodingLengthPerConnection());
        }
        int b = m.getBond(zi,zj);

        BitSet bs = new BitSet(4 + 3);
        if(m.getBondTypeSimple(b)==Molecule.cBondTypeSingle){bs.set(0);}
        if(m.getBondTypeSimple(b)==Molecule.cBondTypeDouble){bs.set(1);}
        if(m.getBondTypeSimple(b)==Molecule.cBondTypeTriple){bs.set(2);}
        if(m.getBondTypeSimple(b)==Molecule.cBondTypeDelocalized){bs.set(3);}

        if(m.getBondTypeSimple(b)==Molecule.cBondTypeSingle){
            if( m.getBondType(b) == Molecule.cBondTypeUp ) {bs.set(4);}
            if( m.getBondType(b) == Molecule.cBondTypeDown ) {bs.set(5);}
            if( m.getBondType(b) == Molecule.cBondTypeCross ) {bs.set(6);}
        }

        return bs;
    }

    public static BitSet oneHot(int set, int length) {
        BitSet bsi = new BitSet(length);
        bsi.set(set);
        return bsi;
    }
    public static BitSet multiHot(int set[], int length) {
        BitSet bsi = new BitSet(length);
        Arrays.stream(set).forEach(xi -> bsi.set(xi));
        return bsi;
    }

    public static void print(List<BitSet> bs, int lengthPerBitSet) {
        for(BitSet bsi : bs) {
            System.out.print("\n");
            for(int zi=0;zi<lengthPerBitSet;zi++) {
                System.out.print( bsi.get(zi)?"1":0 );
            }
        }
    }

    public static void main(String args[]) {
        StereoMolecule mi = ChemUtils.parseSmiles("C(=C/c1cccc2ccccc12)\\c1cc[nH+]cc1");

        MolEncoder ei = new MolEncoder();
        ei.setMolecule(mi);

        List<BitSet> encoded = ei.encodeMolecule();
        print(encoded,ei.getAtomEncodingLength()+ei.N*ei.getConnectionEncodingLengthPerConnection());
        System.out.println("");
        //for(BitSet bi : encoded) {
        //    System.out.println(bi.toString());
        //}
    }


}
