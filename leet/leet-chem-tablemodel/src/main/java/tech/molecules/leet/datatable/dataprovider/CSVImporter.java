package tech.molecules.leet.datatable.dataprovider;

import com.actelion.research.chem.IDCodeParser;
import com.actelion.research.chem.SmilesParser;
import com.actelion.research.chem.StereoMolecule;
import org.apache.commons.lang3.tuple.Pair;
import org.hibernate.cache.spi.access.CollectionDataAccess;
import tech.molecules.leet.chem.StructureWithID;
import tech.molecules.leet.datatable.DataProvider;
import tech.molecules.leet.datatable.DataTable;
import tech.molecules.leet.datatable.DataTableColumn;
import tech.molecules.leet.datatable.chem.DefaultStructureProvider;
import tech.molecules.leet.datatable.chem.StructurePropertiesProvider;
import tech.molecules.leet.datatable.chem.StructureWithIDColumn;
import tech.molecules.leet.datatable.column.NumericColumn;
import tech.molecules.leet.datatable.column.StringColumn;

import java.io.*;
import java.util.*;

public class CSVImporter {

    private File file;

    public enum ColType {String,Numeric,Idcode,Smiles}

    /**
     * Header format:
     *
     * colname[..],colname2,colname[..]
     *
     * in brackets it is possible to specify a type:
     * supported options are [numeric] and [idcode]
     * in the future things like multi-numeric etc might be added.
     *
     * @param csvWithHeader
     */
    public CSVImporter(File csvWithHeader) {
        this.file = csvWithHeader;
        parseHeader();
    }

    private List<ColType> columns = new ArrayList<>();

    private void parseHeader() {
        try(BufferedReader in = new BufferedReader(new FileReader(file))) {
            String header = in.readLine();
            String[] splits = header.split(",");
            for(int zi=0;zi<splits.length;zi++) {
                if( splits[zi].trim().toLowerCase().endsWith("[numeric]")) {
                    columns.add(ColType.Numeric);
                }
                else if ( splits[zi].trim().toLowerCase().endsWith("[idcode]")) {
                    columns.add(ColType.Numeric);
                }
                else if ( splits[zi].trim().toLowerCase().endsWith("[smiles]")) {
                    columns.add(ColType.Smiles);
                }
                else {
                    columns.add(ColType.String);
                }
            }
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static class ImportedCSV {
        private List<ColType> columns;
        private List<Pair<DataProvider, DataTableColumn>> tableData;
        private List<String> keys;

        public ImportedCSV(List<ColType> columns, List<Pair<DataProvider, DataTableColumn>> tableData, List<String> keys) {
            this.columns = columns;
            this.tableData = tableData;
            this.keys = keys;
        }

        public List<ColType> getColumns() {
            return columns;
        }

        public List<Pair<DataProvider, DataTableColumn>> getTableData() {
            return tableData;
        }

        public List<String> getKeys() {
            return keys;
        }
    }

    public static DataTable createDataTable(ImportedCSV data) {
        DataTable dt = new DataTable();
        for(Pair<DataProvider,DataTableColumn> ci : data.tableData) {
            ci.getRight().setDataProvider(ci.getLeft());
            dt.addDataColumn(ci.getRight());
        }
        return dt;
    }

    /**
     *
     * @param colForId can be -1, then generic ids are generated ("0000000","0000001"..)
     * @return
     */
    public ImportedCSV createTableData(int colForId) {
        List<Pair<DataProvider, DataTableColumn>> data = new ArrayList<>();

        for (int zi = 0; zi < columns.size(); zi++) {
            switch (columns.get(zi)) {
                case Idcode:
                case Smiles:
                {
                    data.add(Pair.of(new DefaultStructureProvider(), new StructureWithIDColumn()));
                    break;
                }
                case Numeric:
                    data.add(Pair.of(new DefaultNumericDataProvider(new HashMap<>()), new NumericColumn()));
                    break;
                case String:
                    data.add(Pair.of(new HashMapBasedDataProvider(new HashMap()), new StringColumn()));
            }
        }

        List<String> keys;
        try (BufferedReader in = new BufferedReader(new FileReader(file))) {
            String header = in.readLine();

            int cnt = 0;
            String line = null;

            StereoMolecule mi = new StereoMolecule();
            IDCodeParser icp = new IDCodeParser();
            keys = new ArrayList<>();
            while ((line = in.readLine()) != null) {
                String splits[] = line.split(",");

                String id_a = String.format("%07d", cnt);
                if (colForId >= 0) {
                    id_a = splits[colForId];
                }

                keys.add(id_a);

                int cc = 0;
                for (ColType ci : columns) {
                    if(splits.length<=cc) {
                        continue;
                    }

                    switch (ci) {
                        case Idcode:
                            try {
                                icp.parse(mi, splits[cc]);
                            } catch (Exception ex) {
                                mi.clear();
                                System.out.println("[WARN] parsing error..");
                            }
                            ((DefaultStructureProvider) data.get(cc).getLeft()).
                                    //loadStructures(Collections.singletonList(new StructureWithID(id_a,"", new String[] {mi.getIDCode(),mi.getIDCoordinates()})));
                                            loadStructures(Collections.singletonList(Pair.of(id_a, new String[]{mi.getIDCode(), mi.getIDCoordinates()})),false);
                            break;
                        case Smiles:
                            try {
                                new SmilesParser().parse(mi, splits[cc]);
                            } catch (Exception ex) {
                                mi.clear();
                                System.out.println("[WARN] parsing error..");
                            }
                            ((DefaultStructureProvider) data.get(cc).getLeft()).
                                    //loadStructures(Collections.singletonList(new StructureWithID(id_a,"", new String[] {mi.getIDCode(),mi.getIDCoordinates()})));
                                            loadStructures(Collections.singletonList(Pair.of(id_a, new String[]{mi.getIDCode(), mi.getIDCoordinates()})),false);
                            break;
                        case Numeric:
                            double di = Double.NaN;
                            try {
                                di = Double.parseDouble(splits[cc]);
                            } catch (NumberFormatException e) {
                                //throw new RuntimeException(e);
                                System.out.println("[WARN] could not parse numeric");
                            }
                            ((DefaultNumericDataProvider) data.get(cc).getLeft()).
                                    addData(Collections.singletonMap(id_a, di));
                            break;
                        case String:
                            ((HashMapBasedDataProvider<String>) data.get(cc).getLeft()).
                                    addData(Collections.singletonMap(id_a, splits[cc]));
                    }
                    cc++;
                }
                cnt++;
            }

        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        return new ImportedCSV(columns, data, keys);
    }

}
