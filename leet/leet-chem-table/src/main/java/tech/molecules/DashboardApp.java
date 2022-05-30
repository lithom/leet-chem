package tech.molecules;

import org.apache.commons.lang3.tuple.Pair;
import tech.molecules.leet.io.CSVIterator;
import tech.molecules.leet.table.*;
import tech.molecules.leet.table.chem.NexusChemPropertiesFilter;
import tech.molecules.leet.table.gui.InteractiveJTable;

import javax.swing.*;
import java.awt.*;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Hello world!
 *
 */
public class DashboardApp
{
    public static void main( String[] args )
    {

        List<Integer> fields = new ArrayList<>(); fields.add(0); fields.add(1);
        CSVIterator iter = null;
        try {
            iter = new CSVIterator("C:\\Temp\\leet\\chembl_structures_short.csv",true, fields);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        int cnt = 0;
        List<Pair<String,String>> molecules = new ArrayList<>();
        while( iter.hasNext() && cnt<20000 ) { //&& cnt < 15000) {
            List<String> di = iter.next();
            molecules.add(Pair.of(di.get(0),di.get(1)));
            cnt++;
        }

        initWithStructures(molecules);
    }

    public static void initWithStructures(List<Pair<String,String>> structuresWithIDs) {
        List<String> all_rows = structuresWithIDs.stream().map( pi -> pi.getLeft() ).collect(Collectors.toList());
        DefaultStructureDataProvider dataprovider = new DefaultStructureDataProvider( structuresWithIDs.stream().map( si -> new String[]{si.getLeft(),si.getRight()} ).collect(Collectors.toList()) );

        JFrame fi = new JFrame();
        fi.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);

        JSplitPane jsp_main = new JSplitPane(JSplitPane.VERTICAL_SPLIT);
        JPanel jp_bottom    = new JPanel();

        fi.getContentPane().setLayout(new BorderLayout());
        fi.getContentPane().add(jsp_main,BorderLayout.CENTER);

        NexusTableModel model = new NexusTableModel();
        List<Pair<NColumn, NDataProvider>> columns = new ArrayList<>();
        StructureColumn col_s = new StructureColumn();
        columns.add(Pair.of( col_s , dataprovider ));
        model.setNexusColumnsWithDataProviders( columns );
        model.setAllRows(all_rows);

        ///InteractiveJTable jtable = new InteractiveJTable();
        NexusTable ntable = new NexusTable(model);

        JScrollPane jsp_table = new JScrollPane(ntable);
        jsp_main.setTopComponent(jsp_table);

        NColumn.NexusRowFilter<NStructureDataProvider> filter_s = new NexusChemPropertiesFilter( (NColumn) col_s );
        model.addRowFilter(col_s,filter_s);

        jsp_main.setBottomComponent(jp_bottom);
        jp_bottom.setLayout(new BorderLayout());
        jp_bottom.add(filter_s.getFilterGUI(),BorderLayout.CENTER);

        fi.setSize(800,600);
        fi.setVisible(true);
    }

}
