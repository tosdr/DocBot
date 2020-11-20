import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*you)(?=.*info)((?=.*promotion)|(?=.*sweepstake)|(?=.*contest)))", "i"),
	caseID: 211
} as Regex;