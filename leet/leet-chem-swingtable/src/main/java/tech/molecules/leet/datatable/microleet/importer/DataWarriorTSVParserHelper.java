package tech.molecules.leet.datatable.microleet.importer;

import tech.molecules.leet.datatable.microleet.model.SerializerMultiNumeric;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.AbstractMap.SimpleEntry;

public class DataWarriorTSVParserHelper {

//    public static void main(String[] args) {
//        String filePath = "path_to_your_tsv_file.tsv";
//        List<SimpleEntry<String, Function<String, Boolean>>> validators = List.of(
//                // add your validators here. e.g.
//                // new SimpleEntry<>("datatype1", str -> str.matches("regex_for_datatype1")),
//                // new SimpleEntry<>("datatype2", str -> str.matches("regex_for_datatype2"))
//        );
//
//        try {
//            List<String> columnTypes = parseFile(filePath, validators);
//            columnTypes.forEach(System.out::println);
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//    }

    public static List<String> getColumnTypesDefault(String file) throws IOException {
        SerializerMultiNumeric sa = new SerializerMultiNumeric();
        List<SimpleEntry<String, Function<String, Boolean>>> validators = new ArrayList<>();

        validators.add(
                new SimpleEntry<>("multinumeric", new Function<String,Boolean>(){
                    @Override
                    public Boolean apply(String str) {
                        return sa.initFromString(str)!=null;
                    }
                }));

        return parseFile(file, validators);
    }

    public static List<String> parseFile(String filePath, List<SimpleEntry<String, Function<String, Boolean>>> validators) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String headerLine = reader.readLine();
            if (headerLine == null) {
                throw new IOException("File is empty!");
            }

            String[] headers = headerLine.split("\t");
            String[] dataValues = new String[headers.length];

            for (int i = 0; i < 20; i++) {
                String dataLine = reader.readLine();
                if (dataLine == null) {
                    break;
                }

                String[] dataLineValues = dataLine.split("\t");
                for (int j = 0; j < dataLineValues.length; j++) {
                    if (i == 0) {
                        dataValues[j] = dataLineValues[j];
                    } else {
                        dataValues[j] += "\t" + dataLineValues[j];
                    }
                }
            }

            List<String> columnTypes = new ArrayList<>();
            for (int i = 0; i < headers.length; i++) {
                columnTypes.add(computeColumnType(headers[i], dataValues[i], validators));
            }

            return columnTypes;
        }
    }

    public static String computeColumnType(String header, String data, List<SimpleEntry<String, Function<String, Boolean>>> validators) {
        if (header.endsWith("[idcode]")) {
            return "chem_structure";
        }

        String[] values = data.split("\t");
        for (SimpleEntry<String, Function<String, Boolean>> validator : validators) {
            boolean allMatch = true;
            for (String value : values) {
                if (!validator.getValue().apply(value)) {
                    allMatch = false;
                    break;
                }
            }

            if (allMatch) {
                return validator.getKey();
            }
        }

        return "no_match";
    }
}
