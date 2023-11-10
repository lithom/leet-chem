package tech.molecules.leet.datatable.chem;

import com.actelion.research.chem.StereoMolecule;
import tech.molecules.leet.chem.ChemUtils;
import tech.molecules.leet.chem.StructureRecord;
import tech.molecules.leet.chem.StructureWithID;
import tech.molecules.leet.chem.mutator.properties.ChemPropertyCounts;
import tech.molecules.leet.datatable.DataFilter;
import tech.molecules.leet.datatable.DataProvider;
import tech.molecules.leet.datatable.NumericDatasource;
import tech.molecules.leet.datatable.column.AbstractDataTableColumn;

import java.util.ArrayList;
import java.util.List;

public class StructureRecordColumn extends AbstractDataTableColumn<StructureRecord,StructureRecord> {

    public StructureRecordColumn() {
        super(StructureRecord.class);
    }

    public StructureRecordColumn(Class<StructureRecord> representationClass, DataProvider<StructureRecord> dp) {
        super(representationClass, dp);
    }

    @Override
    public List<NumericDatasource> getNumericDatasources() {
        List<NumericDatasource> nds = new ArrayList<>();
        for(int zi=0;zi<ChemPropertyCounts.COUNTS_ALL.length;zi++) {
            nds.add(new CalculatedBasicStructurePropertyNumericDatasource(this,zi));
        }
        return nds;
    }

    @Override
    public List<DataFilter<StructureRecord>> getFilters() {
        List<DataFilter<StructureRecord>> filters = new ArrayList<>();
        filters.add(new SubstructureFilter());
        return filters;
    }

    @Override
    public StructureRecord processData(StructureRecord data) {
        //return new StructureWithID(data.molid,data.batchid,data.structure);
        return data;
    }

}
