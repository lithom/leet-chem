package tech.molecules.leet.workbench;


import org.jfree.chart.ChartHints;
import tech.molecules.leet.table.NexusTable;
import tech.molecules.leet.table.gui.JFilterPanel;

import javax.swing.*;
import java.awt.*;
import java.awt.font.TextHitInfo;

/**
 *
 * Provides simple GUI around the table component and offers basic functionality for data analysis
 * and functionality for handling chemical data.
 *
 */
public class JWorkbench extends JPanel {
    private WorkbenchModel model;




    private JPanel jp_Table;
    private JScrollPane jsp_Table;
    private JPanel jp_Filter;

    private NexusTable nt;

    public JWorkbench(WorkbenchModel model) {
        this.model = model;
        this.nt    = new NexusTable(this.model.getNexusTableModel());
        reinit();
    }

    private void reinit() {
        this.removeAll();
        this.setLayout(new BorderLayout());

        this.jp_Table  = new JPanel();
        this.jp_Filter = new JPanel();

        this.jp_Table.setLayout(new BorderLayout());
        this.jp_Filter.setLayout(new BorderLayout());

        this.jsp_Table = new JScrollPane(this.nt);

        this.jp_Table.add(jsp_Table,BorderLayout.CENTER);
        this.jp_Filter.add(nt.getFilterPanel(),BorderLayout.CENTER);
        this.jp_Table.add(nt.getTopPanel(),BorderLayout.NORTH);

        this.add(jp_Table,BorderLayout.CENTER);
        this.add(jp_Filter,BorderLayout.EAST);

    }

}
