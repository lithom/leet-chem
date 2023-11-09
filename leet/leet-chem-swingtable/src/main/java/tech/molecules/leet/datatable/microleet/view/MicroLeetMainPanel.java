package tech.molecules.leet.datatable.microleet.view;

import com.actelion.research.gui.VerticalFlowLayout;
import tech.molecules.leet.datatable.microleet.MicroLeetMain;

import javax.swing.*;
import java.awt.*;
import java.util.function.Supplier;

public class MicroLeetMainPanel extends JPanel {

    private MicroLeetMain main;

    private JPanel panelMainTable;

    private JSplitPane splitPaneLeft;

    private JPanel panelPlots;
    private JPanel panelPlotsTop;
    private JTabbedPane tabbedPanePlots;

    private JPanel panelFilters;
    private JPanel panelFiltersTop;
    private JScrollPane scrollPaneFilters;
    private JPanel panelFiltersArea;


    public MicroLeetMainPanel(MicroLeetMain main) {
        this.main = main;
        reinit();
    }

    private void reinit() {
        reinitLayout();
        initMainTablePanel();
        initPlotsPanel();
        initFilterPanel();
    }

    /**
     * Sets the main layout for panelMainTable, panelPlots and panelFilters.
     */
    private void reinitLayout() {
        this.removeAll();
        this.setLayout(new BorderLayout());
        this.splitPaneLeft = new JSplitPane(JSplitPane.VERTICAL_SPLIT);
        this.panelMainTable = new JPanel();
        this.panelPlots     = new JPanel();
        this.panelFilters   = new JPanel();
        this.splitPaneLeft.setTopComponent(panelMainTable);
        this.splitPaneLeft.setBottomComponent(panelPlots);
        this.add(this.splitPaneLeft,BorderLayout.CENTER);
        this.add(this.panelFilters,BorderLayout.EAST);
    }

    private void initMainTablePanel() {
        this.panelMainTable.removeAll();
        this.panelMainTable.setLayout(new BorderLayout());
        this.panelMainTable.add( main.getSwingTableController() , BorderLayout.CENTER);
    }

    private void initPlotsPanel() {
        this.panelPlots.removeAll();
        this.panelPlots.setLayout(new BorderLayout());
        this.tabbedPanePlots = new JTabbedPane();
        this.panelPlotsTop = new JPanel();
        this.panelPlotsTop.setLayout(new FlowLayout(FlowLayout.LEFT));

        JLabel jl = new JLabel("Plots  ");
        JButton jb_plus = new JButton("+");
        jb_plus.setAction(main.getNewPlotAction());
        this.panelPlotsTop.add(jl);
        this.panelPlotsTop.add(jb_plus);
        jb_plus.setAction(main.getNewPlotAction());
        this.panelPlots.add(panelPlotsTop,BorderLayout.NORTH);
        this.panelPlots.add(tabbedPanePlots,BorderLayout.CENTER);
    }

    private void initFilterPanel() {
        this.panelFilters.removeAll();
        this.panelFilters.setLayout(new BorderLayout());
        this.panelFiltersTop = new JPanel();
        this.panelFiltersTop.setLayout(new FlowLayout(FlowLayout.LEFT));
        JLabel jl = new JLabel("Filters");
        this.panelFiltersTop.add(jl);
        this.panelFiltersTop.setPreferredSize(new Dimension(320,jl.getPreferredSize().height+16));
        this.scrollPaneFilters = new JScrollPane();
        this.panelFiltersArea = new JPanel();
        this.panelFiltersArea.setLayout(new VerticalFlowLayout(VerticalFlowLayout.LEFT,VerticalFlowLayout.TOP));
        this.scrollPaneFilters.setViewportView(this.panelFiltersArea);
        this.panelFilters.add(this.panelFiltersTop,BorderLayout.NORTH);
        this.panelFilters.add(this.scrollPaneFilters,BorderLayout.CENTER);
    }

    public JPanel createAdditionalFilterPanel() {
        if(!SwingUtilities.isEventDispatchThread()) {
            System.out.println("[ERROR] Must be called from EDT..");
        }
        JPanel pi = new JPanel();
        pi.setPreferredSize(new Dimension(240,240));
        panelFiltersArea.add(pi);
        this.scrollPaneFilters.revalidate();
        this.scrollPaneFilters.repaint();
        //SwingUtilities.updateComponentTreeUI(getThis());
        //this.revalidate();
        //this.repaint();
        return pi;
    }

//    public Supplier<JPanel> getFilterPanelSupplier() {
//        Supplier<JPanel> sp = new Supplier<JPanel>() {
//            @Override
//            public JPanel get() {
//                if( !SwingUtilities.isEventDispatchThread()) {
//                    System.out.println("[ERROR] Don't call this from non EDT thread..");
//                }
//                JPanel pi = new JPanel();
//                panelFiltersArea.add(pi);
//                SwingUtilities.updateComponentTreeUI(getThis());
//                return pi;
//            }
//        };
//        return sp;
//    }

    private JPanel getThis() {
        return this;
    }


    public JPanel createAdditionalPlotPanel() {
        if(!SwingUtilities.isEventDispatchThread()) {
            System.out.println("[ERROR] Must be called from EDT..");
        }
        JPanel pi = new JPanel();
        pi.setPreferredSize(new Dimension(240,240));
        this.tabbedPanePlots.addTab("Plot",pi);
        this.tabbedPanePlots.revalidate();
        this.tabbedPanePlots.repaint();
        return pi;
    }
}
