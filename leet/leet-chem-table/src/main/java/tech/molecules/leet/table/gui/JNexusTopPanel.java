package tech.molecules.leet.table.gui;

import tech.molecules.leet.table.NexusTable;
import tech.molecules.leet.table.NexusTableModel;
import tech.molecules.leet.util.ColorMapHelper;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.event.TableModelEvent;
import javax.swing.event.TableModelListener;
import java.awt.*;

public class JNexusTopPanel extends JPanel {

    private NexusTable table;


    private JPanel j_main;
    private JPanel j_left;
    private JPanel j_center;
    private JPanel j_right;


    private JMenuBar jmb;

    private JLabel jl_rowcount;

    private JSlider jsl_size;

    public JNexusTopPanel(NexusTable t) {
        this.table = t;
        this.reinit();
        initListeners();
        this.updateData();
    }

    private void reinit() {
        this.removeAll();
        this.setLayout(new BorderLayout());
        this.j_main = new JPanel();
        j_main.setLayout(new BorderLayout());
        this.add(j_main,BorderLayout.CENTER);
        this.j_left = new JPanel();
        this.j_left.setLayout(new FlowLayout());
        this.j_main.add(this.j_left,BorderLayout.WEST);
        this.j_right = new JPanel();
        this.j_right.setLayout(new FlowLayout());
        this.j_main.add(this.j_right,BorderLayout.EAST);
        this.j_center = new JPanel();
        this.j_center.setLayout(new FlowLayout());
        this.j_main.add(this.j_center,BorderLayout.CENTER);

        this.jmb = new JMenuBar();
        this.j_left.add(this.jmb);

        this.jl_rowcount = new JLabel();
        this.j_right.add(jl_rowcount);

        this.jsl_size = new JSlider(16,400);
        this.jsl_size.addChangeListener(new ChangeListener() {
            @Override
            public void stateChanged(ChangeEvent e) {
                table.setRowHeight(jsl_size.getValue());
            }
        });

        this.j_right.add(jsl_size);
    }

    private void initListeners() {
        table.getModel().addTableModelListener(new TableModelListener() {
            @Override
            public void tableChanged(TableModelEvent e) {
                updateUI();
            }
        });
        ((NexusTableModel)table.getModel()).addNexusListener(new NexusTableModel.NexusTableModelListener() {
            @Override
            public void nexusTableStructureChanged() {
                updateData();
            }
        });
    }

    private void reinitMenu() {
        jmb.removeAll();
        JMenu jm_color = new JMenu("Coloring");
        if(table.getTableModel()!=null) {
            JNumericalDataSourceSelector jndss = new JNumericalDataSourceSelector(new JNumericalDataSourceSelector.NumericalDataSourceSelectorModel(table.getTableModel()), JNumericalDataSourceSelector.SELECTOR_MODE.Menu);
            jm_color.add(jndss.getMenu());
            jndss.addSelectionListener(new JNumericalDataSourceSelector.SelectionListener() {
                @Override
                public void selectionChanged() {
                    // set coloring according to values of nds
                    table.getTableModel().setHighlightColors(ColorMapHelper.evaluateColorValues(null, table.getTableModel(), jndss.getModel().getSelectedDatasource()));
                }
            });
        }
        jmb.add(jm_color);
    }

    public void setSliderManually(int size) {
        jsl_size.setValue(size);
    }

    public void updateData() {
        this.jl_rowcount.setText(String.format("Rows: %d Visible: %d", this.table.getTableModel().getAllRows().size(), this.table.getTableModel().getVisibleRows().size()));
        this.jsl_size.setValue(this.table.getRowHeight());
        this.reinitMenu();
    }
}
