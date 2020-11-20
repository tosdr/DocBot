import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*Google Analytics)|(?=.*analytics)|(?=.*statistics))((?=.*third)|(?=.*party)|(?=.*parties))", "i"),
	caseID: 325
} as Regex;