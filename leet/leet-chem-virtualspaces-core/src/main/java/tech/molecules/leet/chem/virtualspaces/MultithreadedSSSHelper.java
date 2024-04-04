package tech.molecules.leet.chem.virtualspaces;

import com.actelion.research.chem.SSSearcherWithIndex;
import com.actelion.research.chem.StereoMolecule;
import org.apache.commons.lang3.tuple.Pair;

import java.util.HashMap;
import java.util.Map;

public class MultithreadedSSSHelper {

    private Map<Pair<Thread,String>, SSSearcherWithIndex> cachedSSS = new HashMap<>();

    public synchronized SSSearcherWithIndex getSSSForFragment(Thread thread, StereoMolecule fi) {
        String fidc = "";
        synchronized(fi) {
            fidc = fi.getIDCode();
        }
        if(cachedSSS.containsKey( Pair.of(thread,fidc))) {
            return this.cachedSSS.get(Pair.of(thread,fidc));
        }
        else {
            SSSearcherWithIndex searcher = new SSSearcherWithIndex();
            searcher.setFragment(fi,(long[]) null);
            this.cachedSSS.put(Pair.of(thread,fidc),searcher);
            return searcher;
        }
    }

}
