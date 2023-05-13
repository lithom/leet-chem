package tech.molecules.leet.datatable.chem;

import tech.molecules.leet.chem.StructureWithID;
import tech.molecules.leet.chem.mutator.properties.ChemPropertyCounts;
import tech.molecules.leet.datatable.AbstractNumericDatasource;
import tech.molecules.leet.datatable.DataTableColumn;
import tech.molecules.leet.datatable.NumericDatasource;
import tech.molecules.leet.datatable.column.AbstractDataTableColumn;

import java.util.ArrayList;
import java.util.List;

public class StructurePropertiesColumn extends AbstractDataTableColumn<StructurePropertiesProvider.CalculatedBasicStructureProperties,StructurePropertiesProvider.CalculatedBasicStructureProperties> {

    public StructurePropertiesColumn() {
        super(StructurePropertiesProvider.CalculatedBasicStructureProperties.class);
    }

    @Override
    public StructurePropertiesProvider.CalculatedBasicStructureProperties processData(StructurePropertiesProvider.CalculatedBasicStructureProperties data) {
        return data;
    }

    @Override
    public List<NumericDatasource> getNumericDatasources() {
        List<NumericDatasource> datasources = new ArrayList<>();

        for(int zi=0;zi<ChemPropertyCounts.COUNTS_ALL.length;zi++) {
            int fzi = zi;
            ChemPropertyCounts.ChemPropertyCount ci = ChemPropertyCounts.COUNTS_ALL[zi];
            datasources.add(new AbstractNumericDatasource(ci.shortName,getThisColumn()) {

                @Override
                public boolean hasValue(String row) {
                    return true;
                }

                @Override
                public Double evaluate(Object original) {
                    if(original instanceof StructurePropertiesProvider.CalculatedBasicStructureProperties) {
                        StructurePropertiesProvider.CalculatedBasicStructureProperties cbsp = ((StructurePropertiesProvider.CalculatedBasicStructureProperties)original);
                        return (double) cbsp.counts.get(fzi);
                    }
                    return null;
                }

                @Override
                public Class getRepresentationClass() {
                    return Double.class;
                }
            });
        }
        return datasources;
    }
}
