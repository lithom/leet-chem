package tech.molecules.leet.chem;

import com.actelion.research.chem.io.DWARFileParser;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class IOUtils {


    public static List<String> extractSpecialColumnFromDWAR(String filename, String specialColumnName) {
        DWARFileParser dw = new DWARFileParser(filename);
        int spi = dw.getSpecialFieldIndex(specialColumnName);
        List<String> data = new ArrayList<>();
        while(dw.next()) {
            data.add(dw.getSpecialFieldData(spi));
        }
        dw.close();
        return data;
    }

}
