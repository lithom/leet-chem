package tech.molecules.leet.datatable.microleet.model;

import com.actelion.research.chem.StereoMolecule;
import org.apache.commons.lang3.tuple.Pair;
import tech.molecules.leet.chem.StructureRecord;
import tech.molecules.leet.chem.StructureWithID;
import tech.molecules.leet.data.NumericArray;
import tech.molecules.leet.datatable.DataProvider;
import tech.molecules.leet.datatable.DataTable;
import tech.molecules.leet.datatable.DataTableColumn;
import tech.molecules.leet.datatable.chem.StructureColumn;
import tech.molecules.leet.datatable.chem.StructureRecordColumn;
import tech.molecules.leet.datatable.chem.StructureWithIDColumn;
import tech.molecules.leet.datatable.column.NumericArrayColumn;
import tech.molecules.leet.datatable.column.NumericColumn;
import tech.molecules.leet.datatable.column.StringColumn;
import tech.molecules.leet.datatable.swing.DefaultSwingTableModel;

import java.io.BufferedReader;
import java.util.*;

public class MicroLeetDataModel {



    private DataTable T;

    private DefaultSwingTableModel swingTableModel;

    private Map<String,MicroLeetDataType> dataTypes;

    public MicroLeetDataModel() {
        T = new DataTable();
        this.swingTableModel = new DefaultSwingTableModel(this.T);

        this.dataTypes = new HashMap<>();
        initDataTypes();
    }

    private void initDataTypes() {
        MicroLeetDataType<Double> numeric = new MicroLeetDataType<>("numeric", new SerializerNumeric() , new DefaultHashmapBasedDataProviderFactory<>(new SerializerNumeric()) , () -> new NumericColumn() );
        MicroLeetDataType<NumericArray> multinumeric = new MicroLeetDataType<>("multinumeric", new SerializerMultiNumeric() , new DefaultHashmapBasedDataProviderFactory<>(new SerializerMultiNumeric()) , () -> new NumericArrayColumn() );
        MicroLeetDataType<String> string  = new MicroLeetDataType<>("string", new SerializerString() , new DefaultHashmapBasedDataProviderFactory<>(new SerializerString()) , () -> new StringColumn() );
        MicroLeetDataType<StructureRecord> chem  = new MicroLeetDataType<>("chem_structure", new SerializerChemicalStructure() , new DefaultHashmapBasedDataProviderFactory<>(new SerializerChemicalStructure()) , () -> new StructureRecordColumn() );

        this.dataTypes.put(numeric.name,numeric);
        this.dataTypes.put(multinumeric.name,multinumeric);
        this.dataTypes.put(string.name,string);
        this.dataTypes.put(chem.name,chem);
    }

    public MicroLeetDataType<Double> getDataTypeNumeric() { return this.dataTypes.get("numeric"); }
    public MicroLeetDataType<NumericArray> getDataTypeMultiNumeric() { return this.dataTypes.get("multinumeric"); }
    public MicroLeetDataType<String> getDataTypeString() { return this.dataTypes.get("string"); }
    public MicroLeetDataType<StructureRecord> getDataTypeChemStructure() { return this.dataTypes.get("chem_structure"); }

    public DataTable getDataTable() {
        return this.T;
    }

    public DefaultSwingTableModel getSwingTableModel() {
        return this.swingTableModel;
    }

    /**
     *
     * @param lines
     * @param key
     * @param datatypes col_idx -> datatype to parse
     * @param colNames col_idx -> col name
     */
    public void loadCSVFile(Iterator<List<String>> lines, int key, Map<Integer,MicroLeetDataType> datatypes, Map<Integer,String> colNames) {

        Map<Integer, List<Pair<String,String>>> data_raw = new HashMap<>();

        Set<Integer> columnsToParse = new HashSet<>();
        //columnsToParse.add(key);
        columnsToParse.addAll(datatypes.keySet());

        for(Integer ci : columnsToParse) {
            data_raw.put(ci,new ArrayList<>());
        }

        // parse raw data
        Set<String> all_keys = new HashSet<>();
        List<String> all_keys_sorted = new ArrayList<>();

        while(lines.hasNext()) {
            String split[] = lines.next().toArray(new String[0]);
            if( split.length <= key ) {
                System.out.println("[WARN] skip row, no key parsed..");
                continue;
            }
            String ki = split[key];
            if( !all_keys.add(ki) ) {
                System.out.println("[WARN] multiple row key: "+ki);
            }
            else {
                all_keys_sorted.add(ki);
            }
            for(Integer ci : columnsToParse) {
                if(split.length <= ci) {
                    System.out.println("[WARN] skip value, idx too large");
                }
                else {
                    data_raw.get(ci).add( Pair.of(ki, split[ci]));
                }
            }
        }

        // create dataproviders
        for(Integer ci : columnsToParse) {
            DataProvider dp = datatypes.get(ci).dataProviderFactory.initDataProvider(data_raw.get(ci));
            DataTableColumn dtc = datatypes.get(ci).dataColumnFactory.createDataTableColumn();
            dtc.setDataProvider(dp);
            this.T.addDataColumn(dtc);
        }

        this.T.setAllKeys(new ArrayList<>(all_keys));
    }


}
