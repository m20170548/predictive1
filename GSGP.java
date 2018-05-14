package algorithms;

import main.Individual;
import programElements.Addition;
import programElements.Constant;
import programElements.LogisticFunction;
import programElements.Multiplication;
import programElements.Subtraction;
import utils.Data;
import utils.Parameters;
import utils.Utils;

public class GSGP extends GP {

	private static final long serialVersionUID = 7L;

	protected double mutationStep;
	protected boolean boundMutation;

	public GSGP(Data data) {
		super(data);	// go to constructor for GP(data)
		this.mutationStep = Parameters.MGSGP_MS;
		this.boundMutation = Parameters.MGSGP_APPLY_BOUND;
	}

	protected Individual crossover(Individual p1, Individual p2) {
		if (Parameters.BUILD_INDIVIDUALS)
			return crossoverIndividual(p1, p2);
		else
			return crossoverSemantics(p1, p2);

	}

	protected Individual crossoverIndividual(Individual p1, Individual p2) {

		Individual offspring = new Individual();

		offspring.addProgramElement(new Addition());
		offspring.addProgramElement(new Multiplication());

		for (int i = 0; i < p1.getSize(); i++)
			offspring.addProgramElement(p1.getProgramElementAtIndex(i));

		Individual randomTree = initializer.grow(Parameters.IN_DEPTH_LIM);

		offspring.addProgramElement(new LogisticFunction());
		for (int i = 0; i < randomTree.getSize(); i++)
			offspring.addProgramElement(randomTree.getProgramElementAtIndex(i));

		offspring.addProgramElement(new Multiplication());
		offspring.addProgramElement(new Subtraction());
		offspring.addProgramElement(new Constant(1.0));

		offspring.addProgramElement(new LogisticFunction());
		for (int i = 0; i < randomTree.getSize(); i++)
			offspring.addProgramElement(randomTree.getProgramElementAtIndex(i));

		for (int i = 0; i < p2.getSize(); i++)
			offspring.addProgramElement(p2.getProgramElementAtIndex(i));

		offspring.calculateDepth();
		return offspring;
	}

	protected Individual crossoverSemantics(Individual p1, Individual p2) {

		Individual offspring = new Individual();

		// create a random tree and evaluate it
		Individual randomTree = initializer.grow(Parameters.IN_DEPTH_LIM);
		randomTree.evaluate(data);

		// build training data semantics
		double[] parent1TrainingSemantics = p1.getTrainingOutputs();
		double[] parent2TrainingSemantics = p2.getTrainingOutputs();
		double[] randomTreeTrainingSemantics = randomTree.getTrainingOutputs();
		double[] offspringTrainingSemantics = buildCrossoverOffspringSemantics(parent1TrainingSemantics,
				parent2TrainingSemantics, randomTreeTrainingSemantics);
		offspring.setTrainingOutputs(offspringTrainingSemantics);

		// build unseen data semantics
		double[] parent1UnseenSemantics = p1.getUnseenOutputs();
		double[] parent2UnseenSemantics = p2.getUnseenOutputs();
		double[] randomTreeUnseenSemantics = randomTree.getUnseenOutputs();
		double[] offspringUnseenSemantics = buildCrossoverOffspringSemantics(parent1UnseenSemantics,
				parent2UnseenSemantics, randomTreeUnseenSemantics);
		offspring.setUnseenOutputs(offspringUnseenSemantics);

		// calculate size and depth
		offspring.setSizeOverride(true);
		offspring.setComputedSize(calculateCrossoverOffspringSize(p1, p2, randomTree));
		offspring.setDepth(calculateCrossoverOffspringDepth(p1, p2, randomTree));

		return offspring;
	}

	protected double[] buildCrossoverOffspringSemantics(double[] parent1Semantics, double[] parent2Semantics,
			double[] randomTreeSemantics) {
		double[] offspringSemantics = new double[parent1Semantics.length];
		for (int i = 0; i < offspringSemantics.length; i++) {
			double randomTreeValue = Utils.logisticFunction(randomTreeSemantics[i]);
			offspringSemantics[i] = (parent1Semantics[i] * randomTreeValue)
					+ ((1.0 - randomTreeValue) * parent2Semantics[i]);
		}
		return offspringSemantics;
	}

	protected int calculateCrossoverOffspringSize(Individual p1, Individual p2, Individual randomTree) {
		return p1.getSize() + p2.getSize() + randomTree.getSize() * 2 + 5;
	}

	protected int calculateCrossoverOffspringDepth(Individual p1, Individual p2, Individual randomTree) {
		int largestParentDepth = Math.max(p1.getDepth(), p2.getDepth());
		// "+ 1" because of the bounding function
		return Math.max(largestParentDepth + 2, randomTree.getDepth() + 3 + 1);
	}

	protected Individual mutation(Individual p) {
		if (Parameters.BUILD_INDIVIDUALS)
			return buildMutationIndividual(p);
		else
			return buildMutationSemantics(p);

	}

	protected Individual buildMutationIndividual(Individual p) {	// combine original individual with 2 random trees 
																	// if MGSGP_APPLY_BOUND is set to TRUE then add logistic functions as well
		Individual offspring = new Individual();
		offspring.addProgramElement(new Addition());	// first node is always Addition

		// copy parent to offspring
		for (int i = 0; i < p.getSize(); i++)
			offspring.addProgramElement(p.getProgramElementAtIndex(i));

		offspring.addProgramElement(new Multiplication());
		offspring.addProgramElement(new Constant(mutationStep));		// ?
		offspring.addProgramElement(new Subtraction());

		// create 2 random trees
		Individual randomTree1 = initializer.grow(Parameters.IN_DEPTH_LIM);
		Individual randomTree2 = initializer.grow(Parameters.IN_DEPTH_LIM);

		if (boundMutation)
			offspring.addProgramElement(new LogisticFunction());	// add logistic function to offspring

		// copy random tree 1 to offspring
		for (int i = 0; i < randomTree1.getSize(); i++)
			offspring.addProgramElement(randomTree1.getProgramElementAtIndex(i));

		if (boundMutation)
			offspring.addProgramElement(new LogisticFunction());

		for (int i = 0; i < randomTree2.getSize(); i++)
			offspring.addProgramElement(randomTree2.getProgramElementAtIndex(i));

		offspring.calculateDepth();
		return offspring;
	}

	protected Individual buildMutationSemantics(Individual p) {

		Individual offspring = new Individual();

		// create 2 random trees and evaluate them
		Individual randomTree1 = initializer.grow(Parameters.IN_DEPTH_LIM);
		Individual randomTree2 = initializer.grow(Parameters.IN_DEPTH_LIM);
		randomTree1.evaluate(data);
		randomTree2.evaluate(data);

		// build training data semantics
		double[] parentTrainingSemantics = p.getTrainingOutputs();
		double[] randomTree1TrainingSemantics = randomTree1.getTrainingOutputs();
		double[] randomTree2TrainingSemantics = randomTree2.getTrainingOutputs();
		double[] offspringTrainingSemantics = buildMutationOffspringSemantics(parentTrainingSemantics,
				randomTree1TrainingSemantics, randomTree2TrainingSemantics);
		offspring.setTrainingOutputs(offspringTrainingSemantics);

		// build unseen data semantics
		double[] parentUnseenSemantics = p.getUnseenOutputs();
		double[] randomTree1UnseenSemantics = randomTree1.getUnseenOutputs();
		double[] randomTree2UnseenSemantics = randomTree2.getUnseenOutputs();
		double[] offspringUnseenSemantics = buildMutationOffspringSemantics(parentUnseenSemantics,
				randomTree1UnseenSemantics, randomTree2UnseenSemantics);
		offspring.setUnseenOutputs(offspringUnseenSemantics);

		// calculate size and depth
		offspring.setSizeOverride(true);
		offspring.setComputedSize(calculateMutationOffspringSize(p, randomTree1, randomTree2));
		offspring.setDepth(calculateMutationOffspringDepth(p, randomTree1, randomTree2));

		return offspring;
	}

	protected double[] buildMutationOffspringSemantics(double[] parentSemantics, double[] randomTree1Semantics,
			double[] randomTree2Semantics) {
		double[] offspringSemantics = new double[parentSemantics.length];
		for (int i = 0; i < offspringSemantics.length; i++) {
			double value1 = randomTree1Semantics[i];
			double value2 = randomTree2Semantics[i];
			if (boundMutation) {
				value1 = Utils.logisticFunction(value1);
				value2 = Utils.logisticFunction(value2);
			}
			offspringSemantics[i] = parentSemantics[i] + (mutationStep * (value1 - value2));
		}
		return offspringSemantics;
	}

	protected int calculateMutationOffspringSize(Individual parent, Individual randomTree1, Individual randomTree2) {
		return parent.getSize() + randomTree1.getSize() + randomTree2.getSize() + 4;
	}

	protected int calculateMutationOffspringDepth(Individual parent, Individual randomTree1, Individual randomTree2) {
		int largestRandomTreeDepth = Math.max(randomTree1.getDepth(), randomTree2.getDepth());
		return Math.max(largestRandomTreeDepth + 3, parent.getDepth() + 1);
	}

	public double getMutationStep() {
		return mutationStep;
	}

	public void setMutationStep(double mutationStep) {
		this.mutationStep = mutationStep;
	}

	public void setBoundMutation(boolean boundMutation) {
		this.boundMutation = boundMutation;
	}
}