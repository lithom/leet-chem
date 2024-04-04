package tech.molecules.chem.coredb.cartridge;

import com.actelion.research.chem.SSSearcher;
import com.actelion.research.chem.StereoMolecule;
import com.actelion.research.chem.descriptor.DescriptorHandlerLongFFP512;
import tech.molecules.leet.chem.BitSetUtils;
import tech.molecules.leet.chem.ChemUtils;

import java.util.BitSet;
import java.util.HashMap;

public class PostgresCartridge {

    public static HashMap<String,BitSet> cacheFFP = new HashMap<>();
    public static HashMap<String, SSSearcher> cacheSSSearcher = new HashMap<>();

    public static final int MAX_CACHE_FFP = 10000;
    public static final int MAX_CACHE_SSSEARCHER = 100;

    public static BitSet getCachedFFP(String idc) {
        if(cacheFFP.containsKey(idc)) {
            return cacheFFP.get(idc);
        }
        else {
            // clear cache?
            if(cacheFFP.size()>=MAX_CACHE_FFP) {
                cacheFFP.clear();
            }

            BitSet ffp_i = ffp_bs(idc);
            cacheFFP.put(idc,ffp_i);
            return ffp_i;
        }
    }

    public static SSSearcher getCachedSSSearcher(String idc) {
        if(cacheSSSearcher.containsKey(idc)) {
            return cacheSSSearcher.get(idc);
        }
        else {
            // clear cache?
            if(cacheSSSearcher.size()>=MAX_CACHE_SSSEARCHER) {
                cacheSSSearcher.clear();
            }

            SSSearcher ssi = new SSSearcher();
            StereoMolecule xi = ChemUtils.parseIDCode(idc);
            xi.setFragment(true);
            ssi.setFragment(xi);

            cacheSSSearcher.put(idc,ssi);
            return ssi;
        }
    }

    public static BitSet ffp_bs(String idcode) {
        StereoMolecule mi = ChemUtils.parseIDCode(idcode);
        long[] fp = (new DescriptorHandlerLongFFP512()).createDescriptor(mi);
        return BitSet.valueOf(fp);
    }

    public static byte[] ffp(String idcode) {
        return ffp_bs(idcode).toByteArray();
    }

    public static synchronized boolean sss(String a_idcode, byte[] a_ffp_ba, String b_idcode) {
        BitSet a_ffp = BitSet.valueOf(a_ffp_ba);
        BitSet b_ffp = getCachedFFP(b_idcode);
        // test if b is subset of a:
        boolean fptest = BitSetUtils.test_subset(b_ffp,a_ffp);
        if(!fptest) {
            return false;
        }

        SSSearcher ssi = getCachedSSSearcher(b_idcode);
        StereoMolecule a = ChemUtils.parseIDCode(a_idcode);
        ssi.setMolecule(a);
        return ssi.isFragmentInMolecule(SSSearcher.cDefaultMatchMode);
    }

    public static synchronized boolean sss(String a_idcode, BitSet a_ffp, String b_idcode) {
        BitSet b_ffp = getCachedFFP(b_idcode);
        // test if b is subset of a:
        boolean fptest = BitSetUtils.test_subset(b_ffp,a_ffp);
        if(!fptest) {
            return false;
        }

        SSSearcher ssi = getCachedSSSearcher(b_idcode);
        StereoMolecule a = ChemUtils.parseIDCode(a_idcode);
        ssi.setMolecule(a);
        return ssi.isFragmentInMolecule(SSSearcher.cDefaultMatchMode);
    }

}
