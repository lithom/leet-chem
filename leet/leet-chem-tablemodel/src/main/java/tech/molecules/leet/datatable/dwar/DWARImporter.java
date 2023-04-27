package tech.molecules.leet.datatable.dwar;

import com.actelion.research.chem.io.DWARFileParser;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class DWARImporter {

    private String filePath;

    public DWARImporter(String filePath) {
        this.filePath = filePath;
    }

    private Map<String, List<String>> dataEntries = new HashMap<>();

    public void init() {
        DWARFileParser dwfp = new DWARFileParser(filePath);



    }

}
