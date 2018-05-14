package algorithms;

import java.io.Serializable;
import java.util.Random;
import main.Individual;
import main.Main;
import main.Population;
import utils.Data;
import utils.Parameters;

public class GP implements Serializable {
	// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	// Parameters
	//
	private static final long serialVersionUID = 7L;
	protected int currentGen;
	protected Random r;
	protected Data data;
	protected Initializer initializer;
	protected Individual currentBest;
	protected Population population;
	protected double bloat, avgFit0, avgSize0, overfitting, btp, tbtp;

	public GP(Data data) {
		this.data = data;
		r = new Random();

		initializer = new Initializer(data.getDimensionality());
		population = initializer.rampedHalfAndHalfInitialization(Parameters.EA_PSIZE, Parameters.IN_DEPTH_LIM);
		population.evaluate(data);

		bloat = overfitting = 0;
		avgFit0 = population.getAVGFitness();
		avgSize0 = population.getAVGSize();
		updateCurrentBest();
		btp = currentBest.getUnseenError();
		tbtp = currentBest.getTrainingError();

		printState();
		currentGen = 1;
		addValue();
	}
	// --------------------------------------------------------------------

	// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	// Evolution
	//
	public void search(int numberOfGen) {	
		for (; currentGen <= numberOfGen; currentGen++) {
			Population offspring = new Population();
			while (offspring.getSize() < population.getSize()) {
				Individual p1, newIndividual;
				p1 = tournamentSelection();
				// apply crossover or mutation
				if (r.nextDouble() < Parameters.VAR_XOVER_PROB) {
					Individual p2 = tournamentSelection();
					newIndividual = crossover(p1, p2);
				} else
					newIndividual = mutation(p1);

				if (Parameters.VAR_APPLY_DEPTH_LIM && newIndividual.getDepth() > Parameters.VAR_DEPTH_LIM)
					newIndividual = p1;		// discard the new individual as it is beyond the depth limit
				else
					newIndividual.evaluate(data);

				offspring.addIndividual(newIndividual);
			}
			population = replacement(offspring);
			updateCurrentBest();
			computeBloat();
			computeOverfitting();
			printState();
			addValue();
		}
	}
	// --------------------------------------------------------------------

	// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	// Selection
	//
	protected Individual tournamentSelection() {
		Population tournamentPopulation = new Population(); 	
		int tournamentSize = (int) (Parameters.VAR_TOUR_PR * population.getSize());  //100 individuals initially
		for (int i = 0; i < tournamentSize; i++)
			tournamentPopulation.addIndividual(population.getIndividual(r.nextInt(population.getSize())));
		return tournamentPopulation.getBest();
	}
	// --------------------------------------------------------------------

	// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	// Variation
	//
	protected Individual crossover(Individual p1, Individual p2) {
		int p1CrossoverStart = r.nextInt(p1.getSize());
		int p1ElementsToEnd = p1.countElementsToEnd(p1CrossoverStart);
		int p2CrossoverStart = r.nextInt(p2.getSize());
		int p2ElementsToEnd = p2.countElementsToEnd(p2CrossoverStart);

		Individual offspring = p1.selectiveDeepCopy(p1CrossoverStart, p1CrossoverStart + p1ElementsToEnd - 1);

		for (int i = 0; i < p2ElementsToEnd; i++)
			offspring.addProgramElementAtIndex(p2.getProgramElementAtIndex(p2CrossoverStart + i), p1CrossoverStart + i);

		offspring.calculateDepth();
		return offspring;
	}

	protected Individual mutation(Individual p) {
		int mutationPoint = r.nextInt(p.getSize());
		int parentElementsToEnd = p.countElementsToEnd(mutationPoint);
		Individual offspring = p.selectiveDeepCopy(mutationPoint, mutationPoint + parentElementsToEnd - 1);
		Individual randomTree = initializer.grow(Parameters.IN_DEPTH_LIM);

		for (int i = 0; i < randomTree.getSize(); i++)
			offspring.addProgramElementAtIndex(randomTree.getProgramElementAtIndex(i), mutationPoint + i);

		offspring.calculateDepth();
		return offspring;
	}

	// this replacement implements elitism
	protected Population replacement(Population offspring) {
		Population nextGeneration = new Population();
		Individual bestParent = population.getBest();
		Individual bestOffspring = offspring.getBest();
		Individual elit;

		if (bestParent.getTrainingError() < bestOffspring.getTrainingError())
			elit = bestParent;
		else
			elit = bestOffspring;

		nextGeneration.addIndividual(elit);
		for (int i = 0; i < offspring.getSize(); i++) {
			if (offspring.getIndividual(i).getId() != elit.getId())
				nextGeneration.addIndividual(offspring.getIndividual(i));
		}
		return nextGeneration;
	}
	// --------------------------------------------------------------------

	// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	// GP Measures
	//
	public void computeBloat() {
		double avgSizeG, avgFitG;
		avgSizeG = population.getAVGSize();
		avgFitG = population.getAVGFitness();
		bloat = ((avgSizeG - avgSize0) / avgSize0) / ((avgFit0 - avgFitG) / avgFit0);
	}

	public void computeOverfitting() {
		if (currentBest.getUnseenError() < btp) {
			overfitting = 0;
			btp = currentBest.getUnseenError();
			tbtp = currentBest.getTrainingError();
		} else
			overfitting = Math.abs(currentBest.getTrainingError() - currentBest.getUnseenError())
					- Math.abs(tbtp - btp);

	}
	// --------------------------------------------------------------------

	// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	// Other
	//
	protected void addValue() {
		Main.output[Main.outputCount][0] = Main.currentRun;
		Main.output[Main.outputCount][1] = currentGen;
		Main.output[Main.outputCount][2] = currentBest.getTrainingError();
		Main.output[Main.outputCount][3] = currentBest.getUnseenError();
		Main.output[Main.outputCount][4] = currentBest.getSize();
		Main.output[Main.outputCount][5] = currentBest.getDepth();
		Main.output[Main.outputCount][6] = bloat;
		Main.output[Main.outputCount++][7] = overfitting;
	}

	protected void updateCurrentBest() {
		currentBest = population.getBest();
	}

	protected void printState() {
		if (Parameters.IO_APPLY_PRINT_GEN) {
			System.out.println("\nBest at generation:\t\t" + currentGen);
			System.out.printf("Training error:\t\t%.2f\nUnseen error:\t\t%.2f\nSize:\t\t\t%d\nDepth:\t\t\t%d\n",
					currentBest.getTrainingError(), currentBest.getUnseenError(), currentBest.getSize(),
					currentBest.getDepth());
		}
	}

	public Individual getCurrentBest() {
		return currentBest;
	}

	public Population getPopulation() {
		return population;
	}
}
