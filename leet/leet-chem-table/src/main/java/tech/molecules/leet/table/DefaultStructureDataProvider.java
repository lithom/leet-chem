package tech.molecules.leet.table;

import com.actelion.research.chem.IDCodeParser;
import com.actelion.research.chem.StereoMolecule;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class DefaultStructureDataProvider implements NStructureDataProvider {

    Map<String,StructureWithID> data = new HashMap<>();

    public DefaultStructureDataProvider(List<String[]> data) {
        this.data.clear();
        IDCodeParser icp = new IDCodeParser();
        for(String[] di : data) {
            StereoMolecule mi = new StereoMolecule();
            icp.parse(mi,di[1]);
            this.data.put( di[0], new StructureWithID(di[0],"",new String[]{mi.getIDCode(),mi.getIDCoordinates()})  );
        }
    }

    @Override
    public StructureWithID getStructureData(String rowid) {
        return this.data.get(rowid);
    }

}
